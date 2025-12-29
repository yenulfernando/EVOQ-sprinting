"""
Cloud-Run Safe Pose Pipeline
----------------------------

- NO GUI calls (no cv2.imshow, waitKey, destroyAllWindows, plt.show)
- All outputs are written under /tmp/pose_pipeline (or POSE_BASE_DIR env)
- Full original logic preserved:
  * YOLO person isolation
  * Frame extraction + auto-crop
  * MediaPipe landmarks
  * Kalman smoothing
  * Angle calculation
  * Rule-based pose classification
  * Best frame export
  * Best angles CSV export

Main entry:
    run_full_pipeline(video_path: str) -> dict
Returns:
    {
        "masked_video_path": ...,
        "best_pose_frames_summary_csv": ...,
        "best_pose_angles_summary_csv": ...
    }
"""

import os
import cv2
import time
import shutil
import glob
import csv
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List

# --- Matplotlib in headless (no display) mode ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class PipelineError(Exception):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


# ==========================================================
# GLOBAL BASE DIRECTORY (Cloud Run safe)
# ==========================================================
BASE_DIR = os.getenv("POSE_BASE_DIR", "pose_pipeline")
os.makedirs(BASE_DIR, exist_ok=True)

# ========== SECTION 1: Keep-One-Object Video Compositor ======
# ==============================================================

# ========= USER SETTINGS ======================================
INPUT_VIDEO = None  # will be set by run_full_pipeline(video_path)

SELECTION_METHOD = "roi"    # "roi" or "class"
INIT_FRAME_INDEX = 20       # frame index to show for ROI and to start tracking

# If SELECTION_METHOD == "class":
CLASS_LABEL = "person"      # e.g., "person", "dog", "car", ...

CAMERA_MODE = "auto"        # "auto" | "static" | "moving"
USE_BG_SUB_FOR_STATIC = True

MODEL_SIZE = "yolov8s-seg"  # YOLO model name (without .pt)
MIN_CONF = 0.25
NMS_IOU = 0.5

IOU_SWITCH_THRESH = 0.25
MAX_MISSES = 30
REID_MIN_OVERLAP = 0.05

WARMUP_FRAMES = 10
WARMUP_IOU_THR = 0.12
WARMUP_BEST_SCORE = 0.10
STEADY_BEST_SCORE = 0.15

MASK_SMOOTH_ALPHA = 0.35
MORPH_OPEN = 3
MORPH_CLOSE = 5
FEATHER_PX = 6

KEEP_ENTIRE_BOX = True
BOX_MODE = "fixed_from_init"        # "fixed_from_init" | "tight_to_detection"
BOX_PAD_PX = 20
BOX_CENTER_SMOOTH_ALPHA = 0.5
BOX_SIZE_SMOOTH_ALPHA = 0.4

# All video outputs under BASE_DIR
WRITE_MASKED = os.path.join(BASE_DIR, "output_masked4.mp4")
WRITE_MASK = os.path.join(BASE_DIR, "output_mask.mp4")
SAVE_PNG_SEQUENCE = False
PNG_DIR = os.path.join(BASE_DIR, "frames_rgba")

# No GUI preview in Cloud Run
PREVIEW = False
PREVIEW_INPAINT_COMPARE = False
PAUSE_ON_START = False

ROI_WINDOW_MAX = (1280, 720)
PRINT_EVERY = 30
# ==============================================================


def ensure_dir(path: str, clear=False):
    if clear and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def get_device_str():
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    except Exception:
        return "cpu"


def load_seg_model(model_name: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install ultralytics\n"
            f"Original error: {e}"
        )
    model = YOLO(model_name + ".pt")
    return model


def is_static_camera(cap, sample=20, step=10, thresh=1.0) -> bool:
    total = 0.0
    count = 0
    pos0 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = [min(i * step, length - 1) for i in range(sample)]
    frames = []
    for p in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))))
    for i in range(len(frames) - 1):
        diff = cv2.absdiff(frames[i], frames[i + 1])
        total += float(np.mean(diff))
        count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
    if count == 0:
        return True
    return (total / count) < thresh


def make_video_writer(path, size, fps_in, is_color=True):
    W, H = size
    fps_out = int(round(min(120, max(1, fps_in))))
    trials = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]
    for fourcc_str, ext in trials:
        out_path = os.path.splitext(path)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(out_path, fourcc, fps_out, (W, H), isColor=is_color)
        if vw.isOpened():
            print(f"[INFO] Using {fourcc_str} -> {out_path} @ {fps_out} FPS (isColor={is_color})")
            return vw, out_path, fps_out
        else:
            vw.release()
    raise RuntimeError("Failed to initialize VideoWriter with tested codecs/containers.")


def masks_from_result(result, min_conf=0.25) -> List[Dict[str, Any]]:
    out = []
    if result.masks is None or result.boxes is None:
        return out
    H, W = result.orig_shape
    masks = result.masks.data
    boxes = result.boxes
    try:
        masks_np = masks.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)
    except Exception:
        masks_np = np.array(masks)
        xyxy = np.array(boxes.xyxy).astype(int)
        confs = np.array(boxes.conf)
        clss = np.array(boxes.cls).astype(int)
    for i in range(masks_np.shape[0]):
        if confs[i] < min_conf:
            continue
        m = masks_np[i]
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (m > 0.5).astype(np.uint8)
        x1, y1, x2, y2 = xyxy[i]
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H, y2))
        out.append(
            {"mask": m, "bbox": (x1, y1, x2, y2), "cls": clss[i], "conf": float(confs[i])}
        )
    return out


def iou_box(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def overlap_ratio(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    inter = np.logical_and(a_mask > 0, b_mask > 0).sum()
    denom = max(1, (a_mask > 0).sum())
    return inter / denom


def ema_mask(prev_f: Optional[np.ndarray], new_u8: np.ndarray, alpha: float) -> np.ndarray:
    new_f = new_u8.astype(np.float32)
    if prev_f is None:
        return new_f
    return alpha * new_f + (1.0 - alpha) * prev_f


def postprocess_mask(mask_u8: np.ndarray, feather_px: int, open_k: int, close_k: int) -> np.ndarray:
    m = mask_u8.copy()
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if feather_px > 0:
        blurred = cv2.GaussianBlur((m * 255).astype(np.uint8), (0, 0),
                                   sigmaX=max(1, feather_px / 2))
        return blurred
    return (m * 255).astype(np.uint8)


def inpaint_hole(bg: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    hole = (mask_u8 == 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hole = cv2.dilate(hole, k, iterations=1)
    out = cv2.inpaint(bg, hole, 3, cv2.INPAINT_TELEA)
    return out


def seek_to_frame_strict(cap, target_idx: int, fps: float, max_step_read=10000) -> int:
    target_idx = max(0, target_idx)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if pos != target_idx:
        ok, _ = cap.read()
        if ok:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            return int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    steps = 0
    while pos < target_idx and steps < max_step_read:
        ok, _ = cap.read()
        if not ok:
            break
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        steps += 1
    return pos


def rect_from_center_size(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = x1 + int(round(w))
    y2 = y1 + int(round(h))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def mask_from_rect(rect: Tuple[int, int, int, int], H: int, W: int) -> np.ndarray:
    x1, y1, x2, y2 = rect
    m = np.zeros((H, W), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def auto_select_roi(frame, model, min_conf=0.25, pad_ratio=0.12):
    """
    Automatically selects the ROI around the main person.
    Returns padded ROI + mask + bbox.
    """
    H, W = frame.shape[:2]
    results = model.predict(frame, conf=min_conf, verbose=False)
    instances = masks_from_result(results[0], min_conf=min_conf)
    persons = [ins for ins in instances if ins["cls"] == 0]
    if not persons:
        raise PipelineError("No sprinter detected clearly in the video.")

    largest = max(persons, key=lambda d: (d["bbox"][2] - d["bbox"][0]) *
                                         (d["bbox"][3] - d["bbox"][1]))
    x1, y1, x2, y2 = largest["bbox"]
    w = x2 - x1
    h = y2 - y1
    pad = int(max(w, h) * pad_ratio)

    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(W, x2 + pad)
    ry2 = min(H, y2 + pad)

    print(f"[AUTO-ROI] Selected: {(x1, y1, x2, y2)} → With padding: {(rx1, ry1, rx2, ry2)}")
    return (rx1, ry1, rx2, ry2), largest["cls"], largest["mask"], largest["bbox"]


def main_yolo_keep_one_object() -> Dict[str, Any]:
    """
    Runs Keep-One-Object compositor.
    Returns a dict with the actual output paths for masked video and mask video.
    """
    if INPUT_VIDEO is None:
        raise RuntimeError("INPUT_VIDEO is not set")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise PipelineError(
            "Invalid video input. Please upload a valid sprint video."
        )

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {INPUT_VIDEO} ({W}x{H} @ {FPS:.2f} FPS), frames={N_FRAMES}")

    cam_mode = CAMERA_MODE
    if cam_mode == "auto":
        cam_static = is_static_camera(cap)
        cam_mode = "static" if cam_static else "moving"
    print(f"[INFO] Camera mode: {cam_mode}")

    device_str = get_device_str()
    print(f"[INFO] Loading model: {MODEL_SIZE} on {device_str}")
    model = load_seg_model(MODEL_SIZE)
    predict_kwargs = dict(iou=NMS_IOU, conf=MIN_CONF, verbose=False,
                          imgsz=max(640, ((max(W, H) + 31) // 32) * 32))

    bg_sub = None
    if cam_mode == "static" and USE_BG_SUB_FOR_STATIC:
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=16, detectShadows=False
        )
        print("[INFO] MOG2 background subtractor enabled (static camera).")

    def get_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return ok, frame

    base_idx = max(0, min(INIT_FRAME_INDEX, N_FRAMES - 1))
    print(f"[INFO] INIT_FRAME_INDEX = {INIT_FRAME_INDEX} (selection frame)")
    ok, init_frame = get_frame(base_idx)
    if not ok:
        raise RuntimeError(f"Could not read frame {base_idx} for initialization.")

    init_results = model.predict(init_frame, **predict_kwargs)
    init_instances = masks_from_result(init_results[0], min_conf=MIN_CONF)
    if len(init_instances) == 0:
        raise PipelineError(
            "No sprinter detected in the video. Please ensure the athlete is clearly visible."
        )


    class_id_to_name = getattr(model.model, "names", getattr(model, "names", {}))
    name_to_class_id = {v: k for k, v in class_id_to_name.items()}

    target_cls_id = None
    target_mask = None
    target_bbox = None
    target_id_text = None

    init_roi_box = None
    box_cx_f = None
    box_cy_f = None
    box_w_f = None
    box_h_f = None

    if SELECTION_METHOD == "class":
        if CLASS_LABEL not in name_to_class_id:
            raise RuntimeError(
                f"CLASS_LABEL '{CLASS_LABEL}' not in model classes. "
                f"Available: {sorted(set(class_id_to_name.values()))[:20]} ..."
            )
        wanted_id = name_to_class_id[CLASS_LABEL]
        cands = [ins for ins in init_instances if ins["cls"] == wanted_id]
        if not cands:
            raise RuntimeError(
                f"No '{CLASS_LABEL}' found on frame {base_idx}. Try another frame or label."
            )
        pick = max(cands, key=lambda d: d["conf"])
        target_cls_id = wanted_id
        target_mask = pick["mask"]
        target_bbox = pick["bbox"]
        target_id_text = f"{CLASS_LABEL} (init conf={pick['conf']:.2f})"
        print(f"[INFO] Selected class: {CLASS_LABEL} (id={target_cls_id}), conf={pick['conf']:.2f}")

        if KEEP_ENTIRE_BOX:
            x1, y1, x2, y2 = target_bbox
            w0 = max(1, (x2 - x1) + 2 * BOX_PAD_PX)
            h0 = max(1, (y2 - y1) + 2 * BOX_PAD_PX)
            cx0 = (x1 + x2) / 2
            cy0 = (y1 + y2) / 2
            box_w_f, box_h_f = float(w0), float(h0)
            box_cx_f, box_cy_f = float(cx0), float(cy0)

    elif SELECTION_METHOD == "roi":
        print("[INFO] AUTO-ROI mode enabled — selecting runner automatically")
        auto_roi_box, cls_id, mask, bbox = auto_select_roi(init_frame, model)

        init_roi_box = auto_roi_box
        target_cls_id = cls_id
        target_mask = mask
        target_bbox = bbox
        target_id_text = "person (auto-selected)"

        rx1, ry1, rx2, ry2 = init_roi_box
        w0 = rx2 - rx1
        h0 = ry2 - ry1
        cx0 = (rx1 + rx2) / 2.0
        cy0 = (ry1 + ry2) / 2.0

        # ROI scale tweak (as in your original)
        scale_w = 1.20
        scale_h = 0.95

        new_w = int(w0 * scale_w)
        new_h = int(h0 * scale_h)

        rx1 = int(cx0 - new_w / 2)
        rx2 = int(cx0 + new_w / 2)
        ry1 = int(cy0 - new_h / 2)
        ry2 = int(cy0 + new_h / 2)

        rx1 = max(0, rx1)
        ry1 = max(0, ry1)
        rx2 = min(W, rx2)
        ry2 = min(H, ry2)

        w0 = new_w
        h0 = new_h

        box_w_f = float(w0)
        box_h_f = float(h0)
        box_cx_f = float(cx0)
        box_cy_f = float(cy0)

        print(f"[INFO] ROI expanded -> width={w0}, height={h0}")

    else:
        raise RuntimeError("SELECTION_METHOD must be 'roi' or 'class'.")

    print(f"[INFO] Will start tracking from frame {base_idx}")

    prev_mask_f = None
    last_assoc_mask = (target_mask > 0).astype(np.uint8)
    last_bbox = target_bbox
    misses = 0

    t0 = time.time()
    frame_counter = 0

    vw_masked, masked_path, FPS_OUT = make_video_writer(WRITE_MASKED, (W, H), FPS, is_color=True)
    vw_mask, mask_path, _ = make_video_writer(WRITE_MASK, (W, H), FPS, is_color=False)

    start_pos = seek_to_frame_strict(cap, base_idx, FPS)
    print(f"[INFO] Decode will start at frame {start_pos} (requested {base_idx})")

    print(f"[INFO] Tracking target: {target_id_text}")
    print(f"[INFO] Writing: {os.path.basename(masked_path)} and {os.path.basename(mask_path)}")

    first_idx_printed = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not first_idx_printed:
            print(f"[INFO] First processed frame index: {idx}")
            first_idx_printed = True

        results = model.predict(frame, **predict_kwargs)
        instances = masks_from_result(results[0], min_conf=MIN_CONF)

        fg_hint = None
        if bg_sub is not None:
            fg = bg_sub.apply(frame)
            fg = cv2.medianBlur(fg, 5)
            _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
            fg_hint = fg

        picked = None
        best_score = -1.0
        for ins in instances:
            if target_cls_id is not None and ins["cls"] != target_cls_id:
                continue
            score = 0.0
            if last_bbox is not None:
                score += 0.7 * iou_box(ins["bbox"], last_bbox)
            if last_assoc_mask is not None:
                inter = overlap_ratio(last_assoc_mask, ins["mask"])
                score += 0.3 * inter
            if fg_hint is not None:
                fg_overlap = overlap_ratio((fg_hint > 0).astype(np.uint8), ins["mask"])
                score += 0.15 * fg_overlap
            if score > best_score:
                best_score = score
                picked = ins

        if picked is None and instances:
            for ins in instances:
                inter = overlap_ratio(last_assoc_mask, ins["mask"]) if last_assoc_mask is not None else 0.0
                if inter > REID_MIN_OVERLAP and inter > best_score:
                    best_score = inter
                    picked = ins

        accepted = False
        relax = (idx - start_pos) < WARMUP_FRAMES
        thr = WARMUP_IOU_THR if relax else IOU_SWITCH_THRESH
        best_thr = WARMUP_BEST_SCORE if relax else STEADY_BEST_SCORE

        if picked is not None:
            if last_bbox is None:
                accepted = True
            else:
                iou = iou_box(picked["bbox"], last_bbox)
                inter = overlap_ratio(last_assoc_mask, picked["mask"]) if last_assoc_mask is not None else 0.0
                if (iou >= thr) or (inter >= thr) or (best_score >= best_thr):
                    accepted = True

        if accepted:
            assoc_cur_mask = (picked["mask"] > 0).astype(np.uint8)
            last_assoc_mask = assoc_cur_mask
            last_bbox = picked["bbox"]
            misses = 0

            if KEEP_ENTIRE_BOX:
                px1, py1, px2, py2 = picked["bbox"]
                cur_cx = (px1 + px2) / 2.0
                cur_cy = (py1 + py2) / 2.0
                if box_cx_f is None:
                    box_cx_f, box_cy_f = float(cur_cx), float(cur_cy)
                else:
                    box_cx_f = (1.0 - BOX_CENTER_SMOOTH_ALPHA) * box_cx_f + BOX_CENTER_SMOOTH_ALPHA * cur_cx
                    box_cy_f = (1.0 - BOX_CENTER_SMOOTH_ALPHA) * box_cy_f + BOX_CENTER_SMOOTH_ALPHA * cur_cy

                if BOX_MODE == "tight_to_detection":
                    cur_w = max(1, (px2 - px1) + 2 * BOX_PAD_PX)
                    cur_h = max(1, (py2 - py1) + 2 * BOX_PAD_PX)
                    if box_w_f is None:
                        box_w_f, box_h_f = float(cur_w), float(cur_h)
                    else:
                        box_w_f = (1.0 - BOX_SIZE_SMOOTH_ALPHA) * box_w_f + BOX_SIZE_SMOOTH_ALPHA * cur_w
                        box_h_f = (1.0 - BOX_SIZE_SMOOTH_ALPHA) * box_h_f + BOX_SIZE_SMOOTH_ALPHA * cur_h
        else:
            misses += 1
            if misses == 1:
                print(f"[WARN] Miss at frame {idx} -> keeping previous mask/box")
            if misses > 0 and picked is not None:
                print(f"[INFO] Re-ID event at frame {idx}: score={best_score:.3f}")
            if misses > MAX_MISSES:
                print(f"[ERR ] Lost target for >{MAX_MISSES} frames. Stopping.")
                break

        if KEEP_ENTIRE_BOX:
            if box_cx_f is None or box_w_f is None:
                x1, y1, x2, y2 = last_bbox
                box_cx_f = (x1 + x2) / 2.0
                box_cy_f = (y1 + y2) / 2.0
                if BOX_MODE == "tight_to_detection":
                    box_w_f = max(1, (x2 - x1) + 2 * BOX_PAD_PX)
                    box_h_f = max(1, (y2 - y1) + 2 * BOX_PAD_PX)
                else:
                    if init_roi_box is not None:
                        rx1, ry1, rx2, ry2 = init_roi_box
                        box_w_f = float(rx2 - rx1)
                        box_h_f = float(ry2 - ry1)
                    else:
                        box_w_f = max(1, (x2 - x1) + 2 * BOX_PAD_PX)
                        box_h_f = max(1, (y2 - y1) + 2 * BOX_PAD_PX)

            rect = rect_from_center_size(box_cx_f, box_cy_f, box_w_f, box_h_f, W, H)
            output_cur_mask = mask_from_rect(rect, H, W)
        else:
            output_cur_mask = last_assoc_mask.copy()

        prev_mask_f = ema_mask(prev_mask_f, output_cur_mask, alpha=MASK_SMOOTH_ALPHA)
        smoothed_u8 = (np.clip(prev_mask_f, 0, 1) * 255).astype(np.uint8)

        post_mask_u8 = postprocess_mask(
            (smoothed_u8 > 127).astype(np.uint8),
            FEATHER_PX, MORPH_OPEN, MORPH_CLOSE
        )

        if post_mask_u8.shape[:2] != frame.shape[:2]:
            post_mask_u8 = cv2.resize(
                post_mask_u8,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        alpha = post_mask_u8.astype(np.float32) / 255.0
        alpha3 = np.dstack([alpha, alpha, alpha])
        out_masked = (frame.astype(np.float32) * alpha3).astype(np.uint8)

        vw_masked.write(out_masked)
        vw_mask.write(post_mask_u8)

        if SAVE_PNG_SEQUENCE:
            ensure_dir(PNG_DIR, clear=False)
            b, g, r = cv2.split(frame)
            a = post_mask_u8
            rgba = cv2.merge([b, g, r, a])
            cv2.imwrite(os.path.join(PNG_DIR, f"frame_{idx:06d}.png"), rgba)

        frame_counter += 1
        if frame_counter % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            fps = frame_counter / max(1e-6, elapsed)
            print(f"[INFO] Processed {frame_counter} frames @ {fps:.2f} FPS")

    cap.release()
    vw_masked.release()
    vw_mask.release()

    total_time = time.time() - t0
    if frame_counter > 0:
        print(f"[DONE] Wrote '{masked_path}' and '{mask_path}'.")
        print(f"[STATS] Frames={frame_counter}, Avg FPS={frame_counter / max(1e-6, total_time):.2f}")
    else:
        print("[DONE] No frames processed.")

    return {"masked_video_path": masked_path, "mask_video_path": mask_path}


# ==========================================================
# SECTION 2: Frames Extract + Auto-Crop + MediaPipe
# ==========================================================

RAW_DIR = os.path.join(BASE_DIR, "exported_frames_raw")
CROPPED_DIR = os.path.join(BASE_DIR, "exported_frames_cropped2")
OUT_DIR_POSE_OVERLAY = os.path.join(BASE_DIR, "exported_frames_with_mp2")
SKEL_DIR = os.path.join(BASE_DIR, "exported_frames_mp_sketch_only")
OUT_CSV_ANGLES_BY_FRAME = os.path.join(BASE_DIR, "pose_angles_by_frame2.csv")

VIS_THRESH = 0.30
LINE_SCALE = 150
DOT_SCALE = 250
SKEL_RGB = (255, 255, 255)
SWAP_AXES = False


def step2_extract_frames(video_path: str) -> Dict[str, Any]:
    print("\n[STEP 2.1] Extract frames from video ->", video_path)
    ensure_dir(RAW_DIR, clear=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        cv2.imwrite(str(Path(RAW_DIR) / f"frame_{frame_idx:06d}.png"), frame)

    cap.release()

    print(f"Video path       : {video_path}")
    print(f"Resolution (wxh) : {w} x {h}")
    print(f"FPS (metadata)   : {fps:.3f}")
    print(f"Frames extracted : {frame_idx}")
    return {"fps": fps, "w": w, "h": h, "frames": frame_idx}


def auto_crop(img, pad_frac=0.02, min_area_ratio=0.01):
    """
    Find the largest bright/non-black region and crop to its bounding box.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, (0, 0, w, h)

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < (w * h * min_area_ratio):
        return img, (0, 0, w, h)

    x, y, bw, bh = cv2.boundingRect(c)
    pad = int(round(max(bw, bh) * pad_frac))

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)

    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def step2_auto_crop_all():
    print("\n[STEP 2.2] Auto-crop all frames")
    ensure_dir(CROPPED_DIR, clear=True)

    raw_files = sorted(Path(RAW_DIR).glob("*.png"))
    if not raw_files:
        raise RuntimeError(f"No frames found in {RAW_DIR}. Run extraction first.")

    saved = 0
    for i, fp in enumerate(raw_files, start=1):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        cropped, _ = auto_crop(img, pad_frac=0.02)
        cv2.imwrite(str(Path(CROPPED_DIR) / f"crop_{i:06d}.png"), cropped)
        saved += 1

    print(f"Cropped {saved} frames → '{CROPPED_DIR}'")


def step2_extract_landmarks_xy():
    print("\n[STEP 2.3] Extracting 33 MediaPipe landmarks (X,Y) normalized by mid-hip")
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError("Please install MediaPipe first: pip install mediapipe")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    out_csv = os.path.join(BASE_DIR, "pose_landmarks_xy_by_frame.csv")
    files = sorted(Path(CROPPED_DIR).glob("*.png"))
    if not files:
        raise RuntimeError("No cropped frames found. Run the crop step first.")

    landmark_names = [lm.name for lm in mp_pose.PoseLandmark]
    records = []

    for idx, fp in enumerate(files, start=1):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        row = {"FRAME_NAME": fp.name}
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            lh = np.array(
                [
                    lms[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                    lms[mp_pose.PoseLandmark.LEFT_HIP].y * h,
                ]
            )
            rh = np.array(
                [
                    lms[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                    lms[mp_pose.PoseLandmark.RIGHT_HIP].y * h,
                ]
            )
            pelvis = (lh + rh) / 2.0

            for lm in landmark_names:
                idx_lm = mp_pose.PoseLandmark[lm].value
                pt = lms[idx_lm]
                if pt.visibility < 0.3:
                    row[f"{lm}_X"] = np.nan
                    row[f"{lm}_Y"] = np.nan
                else:
                    row[f"{lm}_X"] = (pt.x * w) - pelvis[0]
                    row[f"{lm}_Y"] = (pt.y * h) - pelvis[1]
        else:
            for lm in landmark_names:
                row[f"{lm}_X"] = np.nan
                row[f"{lm}_Y"] = np.nan

        records.append(row)

    import pandas as pd

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    pose.close()
    print(f"✅ Saved mid-hip normalized coordinates to {out_csv}")
    return out_csv


def step2_apply_kalman_filter(input_csv):
    print("\n[STEP 2.4] Applying Kalman filter to landmark coordinates")
    import pandas as pd
    from pykalman import KalmanFilter

    df = pd.read_csv(input_csv)
    filled_df = df.copy()

    def apply_kalman(series):
        values = series.to_numpy(dtype=np.float64)
        mask = np.isnan(values)
        if np.all(mask):
            return series
        temp = (
            pd.Series(values)
            .interpolate(method="linear", limit_direction="both")
            .to_numpy()
        )
        transition_matrix = np.array([[1, 1], [0, 1]])
        observation_matrix = np.array([[1, 0]])
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=[temp[0], 0],
            n_dim_obs=1,
        )
        kf = kf.em(temp.reshape(-1, 1), n_iter=5)
        smoothed_state_means, _ = kf.smooth(temp.reshape(-1, 1))
        return pd.Series(smoothed_state_means[:, 0], index=series.index)

    for col in df.columns:
        if col.endswith(("_X", "_Y")):
            print(f"Filtering {col} ...")
            filled_df[col] = apply_kalman(df[col])

    out_path = os.path.join(BASE_DIR, "pose_landmarks_xy_kalman_filled.csv")
    filled_df.to_csv(out_path, index=False)
    print(f"✅ Missing values filled with Kalman filtering -> {out_path}")
    return out_path


def step2_draw_skeletons_from_kalman(csv_path):
    print("\n[STEP 2.5] Drawing skeletons from Kalman-filtered data")

    import pandas as pd

    out_dir = os.path.join(BASE_DIR, "pose_clean_full_skeleton_black")
    ensure_dir(out_dir, clear=True)

    df = pd.read_csv(csv_path)
    frames = len(df)

    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 31),
        (24, 26), (26, 28), (28, 32),
        (27, 29), (28, 30),
        (15, 17), (16, 18), (17, 19), (18, 20),
        (19, 21), (20, 22)
    ]

    landmark_names = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    ]

    all_x, all_y = [], []
    for n in landmark_names:
        all_x.extend(df[f"{n}_X"].dropna().tolist())
        all_y.extend(df[f"{n}_Y"].dropna().tolist())

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    pad_x = (xmax - xmin) * 0.20
    pad_y = (ymax - ymin) * 0.20
    xmin_fixed, xmax_fixed = xmin - pad_x, xmax + pad_x
    ymin_fixed, ymax_fixed = ymin - pad_y, ymax + pad_y

    print("✅ Global axis range fixed for all frames:")
    print(f"X: {xmin_fixed:.2f} to {xmax_fixed:.2f}")
    print(f"Y: {ymin_fixed:.2f} to {ymax_fixed:.2f}")

    for i in range(frames):
        fig, ax = plt.subplots(figsize=(2.4, 5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        points = np.array(
            [[df.loc[i, f"{n}_X"], df.loc[i, f"{n}_Y"]] for n in landmark_names]
        )
        for a, b in connections:
            if not np.isnan(points[a, 0]) and not np.isnan(points[b, 0]):
                ax.plot(
                    [points[a, 0], points[b, 0]],
                    [points[a, 1], points[b, 1]],
                    color="deepskyblue",
                    linewidth=2.5,
                )
        ax.scatter(points[:, 0], points[:, 1], color="yellow", s=15)
        ax.scatter(0, 0, color="lime", s=40)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(xmin_fixed, xmax_fixed)
        ax.set_ylim(ymax_fixed, ymin_fixed)

        out_path = os.path.join(out_dir, f"frame_{i+1:04d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0, facecolor="black")
        plt.close(fig)

    print(f"✅ Saved {frames} skeleton frames to '{out_dir}'")
    return out_dir


def step2_calculate_angles_from_kalman(csv_path):
    print("\n[STEP 2.6] Calculating joint angles from Kalman-smoothed data")
    import pandas as pd

    df = pd.read_csv(csv_path)
    out_csv = os.path.join(BASE_DIR, "pose_angles_from_kalman.csv")

    def angle(pa, pb, pc):
        if any(np.isnan(pa)) or any(np.isnan(pb)) or any(np.isnan(pc)):
            return np.nan
        ba = pa - pb
        bc = pc - pb
        dot = np.dot(ba, bc)
        mag = np.linalg.norm(ba) * np.linalg.norm(bc)
        if mag == 0:
            return np.nan
        return np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0)))

    joints = {
        "RIGHT_KNEE": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        "LEFT_KNEE": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "RIGHT_HIP": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
        "LEFT_HIP": ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
        "RIGHT_ELBOW": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        "LEFT_ELBOW": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "RIGHT_SHOULDER": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
        "LEFT_SHOULDER": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
        "RIGHT_ANKLE": ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
        "LEFT_ANKLE": ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    }

    out_rows = []
    for i in range(len(df)):
        row = {"FRAME_NAME": i + 1}
        for name, (A, B, C) in joints.items():
            pa = np.array([df.loc[i, f"{A}_X"], df.loc[i, f"{A}_Y"]])
            pb = np.array([df.loc[i, f"{B}_X"], df.loc[i, f"{B}_Y"]])
            pc = np.array([df.loc[i, f"{C}_X"], df.loc[i, f"{C}_Y"]])
            row[name] = angle(pa, pb, pc)
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"✅ Saved angles from Kalman dataset to {out_csv}")
    return out_csv


# ==========================================================
# SECTION 3: Rule-based Classification + Top-K
# ==========================================================

try:
    import mediapipe as mp
except Exception:
    mp = None


def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    ab = [a[0] - b[0], a[1] - b[1]]
    cb = [c[0] - b[0], c[1] - b[1]]

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0.0

    cosang = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    angle = math.degrees(math.acos(cosang))
    return angle


def compute_joint_angles(lm, mp_pose_=None):
    if mp_pose_ is None:
        import mediapipe as mp_
        mp_pose_ = mp_.solutions.pose

    angles = {}
    angles["Right Knee"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.RIGHT_HIP],
        lm[mp_pose_.PoseLandmark.RIGHT_KNEE],
        lm[mp_pose_.PoseLandmark.RIGHT_ANKLE],
    )
    angles["Left Knee"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.LEFT_HIP],
        lm[mp_pose_.PoseLandmark.LEFT_KNEE],
        lm[mp_pose_.PoseLandmark.LEFT_ANKLE],
    )
    angles["Right Hip"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.RIGHT_SHOULDER],
        lm[mp_pose_.PoseLandmark.RIGHT_HIP],
        lm[mp_pose_.PoseLandmark.RIGHT_KNEE],
    )
    angles["Left Hip"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.LEFT_SHOULDER],
        lm[mp_pose_.PoseLandmark.LEFT_HIP],
        lm[mp_pose_.PoseLandmark.LEFT_KNEE],
    )
    angles["Right Elbow"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.RIGHT_SHOULDER],
        lm[mp_pose_.PoseLandmark.RIGHT_ELBOW],
        lm[mp_pose_.PoseLandmark.RIGHT_WRIST],
    )
    angles["Left Elbow"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.LEFT_SHOULDER],
        lm[mp_pose_.PoseLandmark.LEFT_ELBOW],
        lm[mp_pose_.PoseLandmark.LEFT_WRIST],
    )
    angles["Right Shoulder"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.RIGHT_ELBOW],
        lm[mp_pose_.PoseLandmark.RIGHT_SHOULDER],
        lm[mp_pose_.PoseLandmark.RIGHT_HIP],
    )
    angles["Left Shoulder"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.LEFT_ELBOW],
        lm[mp_pose_.PoseLandmark.LEFT_SHOULDER],
        lm[mp_pose_.PoseLandmark.LEFT_HIP],
    )
    angles["Right Ankle (Heel-to-Toe)"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.RIGHT_KNEE],
        lm[mp_pose_.PoseLandmark.RIGHT_ANKLE],
        lm[mp_pose_.PoseLandmark.RIGHT_FOOT_INDEX],
    )
    angles["Left Ankle (Heel-to-Toe)"] = calculate_angle(
        lm[mp_pose_.PoseLandmark.LEFT_KNEE],
        lm[mp_pose_.PoseLandmark.LEFT_ANKLE],
        lm[mp_pose_.PoseLandmark.LEFT_FOOT_INDEX],
    )
    return angles


POSE_LIST = ["Toe Off", "MVP", "Strike", "Touch Down", "Full Support"]


def classify_pose(angle_dict, min_hits=4):
    """
    Compare Kalman-smoothed pose angles against rule-based thresholds.
    Returns continuous confidence (0–100%) based on proximity to ideal ranges.
    Evaluates both right-leg and left-leg versions.
    """
    import numpy as np

    def range_score(value, low, high, tol=10):
        if np.isnan(value):
            return 0.0
        center = (low + high) / 2
        width = (high - low) / 2
        diff = abs(value - center)
        if diff <= width:
            return 1.0
        elif diff <= width + tol:
            return max(0.0, 1 - (diff - width) / tol)
        else:
            return 0.0

    rk = angle_dict.get("RIGHT_KNEE", np.nan)
    lk = angle_dict.get("LEFT_KNEE", np.nan)
    rh = angle_dict.get("RIGHT_HIP", np.nan)
    lh = angle_dict.get("LEFT_HIP", np.nan)
    re = angle_dict.get("RIGHT_ELBOW", np.nan)
    le = angle_dict.get("LEFT_ELBOW", np.nan)
    rs = angle_dict.get("RIGHT_SHOULDER", np.nan)
    ls = angle_dict.get("LEFT_SHOULDER", np.nan)
    ra = angle_dict.get("RIGHT_ANKLE", np.nan)
    la = angle_dict.get("LEFT_ANKLE", np.nan)

    rules_right = {
        "Toe Off": [
            (rk, 75, 85), (lk, 160, 175), (rh, 95, 115), (lh, 160, 175),
            (re, 95, 140), (le, 80, 95), (rs, 55, 75), (ls, 40, 65),
            (ra, 95, 135), (la, 130, 145),
        ],
        "MVP": [
            (rk, 110, 135), (lk, 115, 127), (rh, 150, 175), (lh, 120, 145),
            (re, 47, 67), (le, 130, 150), (rs, 55, 70), (ls, 65, 85),
            (ra, 120, 140), (la, 125, 150),
        ],
        "Strike": [
            (rk, 55, 75), (lk, 150, 165), (rh, 170, 180), (lh, 130, 140),
            (re, 100, 115), (le, 115, 130), (rs, 10, 25), (ls, 30, 45),
            (ra, 140, 170), (la, 90, 110),
        ],
        "Touch Down": [
            (rk, 45, 60), (lk, 155, 170), (rh, 150, 170), (lh, 145, 160),
            (re, 130, 150), (le, 105, 120), (rs, 0, 10), (ls, 10, 20),
            (ra, 130, 160), (la, 85, 105),
        ],
        "Full Support": [
            (rk, 15, 30), (lk, 135, 160), (rh, 130, 145), (lh, 150, 165),
            (re, 155, 175), (le, 100, 130), (rs, 0, 10), (ls, 5, 15),
            (ra, 95, 115), (la, 120, 150),
        ],
    }

    # Mirror for left side
    rules_left = {}
    for pose, joints in rules_right.items():
        mirrored = [
            (joints[1][0], joints[1][1], joints[1][2]),
            (joints[0][0], joints[0][1], joints[0][2]),
            (joints[3][0], joints[3][1], joints[3][2]),
            (joints[2][0], joints[2][1], joints[2][2]),
            (joints[5][0], joints[5][1], joints[5][2]),
            (joints[4][0], joints[4][1], joints[4][2]),
            (joints[7][0], joints[7][1], joints[7][2]),
            (joints[6][0], joints[6][1], joints[6][2]),
            (joints[9][0], joints[9][1], joints[9][2]),
            (joints[8][0], joints[8][1], joints[8][2]),
        ]
        rules_left[pose] = mirrored

    pose_scores_right = {}
    pose_scores_left = {}
    for pose, rules in rules_right.items():
        vals = [range_score(val, lo, hi) for (val, lo, hi) in rules]
        pose_scores_right[pose] = np.nanmean(vals) * 100
    for pose, rules in rules_left.items():
        vals = [range_score(val, lo, hi) for (val, lo, hi) in rules]
        pose_scores_left[pose] = np.nanmean(vals) * 100

    best_right = max(pose_scores_right, key=pose_scores_right.get)
    best_left = max(pose_scores_left, key=pose_scores_left.get)
    right_conf = pose_scores_right[best_right]
    left_conf = pose_scores_left[best_left]

    thresh = (min_hits / 10 * 100)
    if right_conf >= left_conf and right_conf >= thresh:
        return f"{best_right} (Right)", round(right_conf, 2)
    elif left_conf > right_conf and left_conf >= thresh:
        return f"{best_left} (Left)", round(left_conf, 2)
    else:
        return "Unknown Pose", 0.0


def classify_from_kalman_angles(angle_csv, min_hits=4):
    """
    Run rule-based classification directly on Kalman-smoothed angle data.
    """
    import pandas as pd

    df = pd.read_csv(angle_csv)
    results = []

    for i, row in df.iterrows():
        angles = row.to_dict()
        pose, conf = classify_pose(angles, min_hits=min_hits)
        results.append(
            {"FRAME": row.get("FRAME_NAME", i + 1), "POSE": pose, "CONFIDENCE": conf}
        )

    out_df = pd.DataFrame(results)
    out_path = os.path.join(BASE_DIR, "classified_poses_from_kalman.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved classified results to {out_path}")
    return out_path


def visualize_classified_skeletons(
    classified_csv,
    skeleton_folder,
    out_dir=None,
):
    """
    Optional helper to annotate skeletons with pose labels.
    """
    import pandas as pd

    if out_dir is None:
        out_dir = os.path.join(BASE_DIR, "classified_pose_visuals")
    ensure_dir(out_dir, clear=True)

    df = pd.read_csv(classified_csv)

    for i, row in df.iterrows():
        frame = int(row["FRAME"])
        pose = row["POSE"]
        conf = row["CONFIDENCE"]
        img_path = os.path.join(skeleton_folder, f"frame_{frame:04d}.png")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        color = (0, 255, 0) if pose != "Unknown Pose" else (0, 0, 255)
        text = f"{pose} ({conf:.2f}%)"
        cv2.putText(
            img,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA,
        )

        out_path = os.path.join(out_dir, f"{frame:04d}_{pose}.png")
        cv2.imwrite(out_path, img)

    print(f"✅ Visualized classified skeletons saved to: {out_dir}")
    return out_dir


def export_best_pose_frames(
    classified_csv,
    skeleton_folder,
    out_dir=None,
    poses=("Toe Off", "MVP", "Strike", "Touch Down", "Full Support"),
):
    """
    Pick the single highest-confidence frame for each pose.
    """
    import pandas as pd

    if out_dir is None:
        out_dir = os.path.join(BASE_DIR, "best_pose_frames_from_kalman")
    ensure_dir(out_dir, clear=True)

    df = pd.read_csv(classified_csv)
    summary_rows = []

    for pose_name in poses:
        cand = df[df["POSE"].str.contains(pose_name, case=False, na=False)]
        if cand.empty:
            print(f"[WARN] No frames found for pose: {pose_name}")
            continue

        best = cand.loc[cand["CONFIDENCE"].idxmax()]
        frame = int(best["FRAME"])
        conf = float(best["CONFIDENCE"])

        img_path = os.path.join(skeleton_folder, f"frame_{frame:04d}.png")
        if not os.path.exists(img_path):
            print(f"[WARN] Skeleton image missing for frame {frame}: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        label = f"{pose_name} ({conf:.2f}%)"
        color = (0, 255, 0) if pose_name != "Unknown Pose" else (0, 0, 255)
        cv2.putText(
            img,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        out_name = f"{pose_name.replace(' ', '_')}__frame_{frame:04d}__{conf:.1f}.png"
        out_img_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_img_path, img)

        summary_rows.append(
            {
                "POSE": pose_name,
                "FRAME": frame,
                "CONFIDENCE": round(conf, 2),
                "IMAGE_PATH": out_img_path,
            }
        )

    if summary_rows:
        summary_csv = os.path.join(out_dir, "best_pose_frames_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"✅ Best frames saved in: {out_dir}")
        print(f"✅ Summary CSV: {summary_csv}")
        return summary_csv
    else:
        print("[INFO] No best frames exported (no poses found).")
        return None


def visualize_best_pose_angles(
    best_summary_csv,
    kalman_angles_csv,
    out_csv=None,
):
    """
    Extract the 10 rule-based joint angles for the best 5 poses.
    """
    import pandas as pd

    if out_csv is None:
        out_csv = os.path.join(BASE_DIR, "best_pose_angles_summary.csv")

    if not os.path.exists(best_summary_csv):
        raise FileNotFoundError(f"Missing best pose summary: {best_summary_csv}")
    if not os.path.exists(kalman_angles_csv):
        raise FileNotFoundError(f"Missing angle data: {kalman_angles_csv}")

    summary_df = pd.read_csv(best_summary_csv)
    angle_df = pd.read_csv(kalman_angles_csv)

    angle_cols = [
        "RIGHT_KNEE",
        "LEFT_KNEE",
        "RIGHT_HIP",
        "LEFT_HIP",
        "RIGHT_ELBOW",
        "LEFT_ELBOW",
        "RIGHT_SHOULDER",
        "LEFT_SHOULDER",
        "RIGHT_ANKLE",
        "LEFT_ANKLE",
    ]

    records = []
    for _, row in summary_df.iterrows():
        pose = row["POSE"]
        frame = int(row["FRAME"])
        conf = row["CONFIDENCE"]
        img_path = row.get("IMAGE_PATH", "")

        if "FRAME_NAME" in angle_df.columns:
            match = angle_df[angle_df["FRAME_NAME"] == frame]
        else:
            match = angle_df.iloc[[frame - 1]] if frame - 1 < len(angle_df) else None

        if match is None or match.empty:
            print(f"[WARN] No angle data for frame {frame}")
            continue

        angles = match.iloc[0]
        record = {"POSE": pose, "FRAME": frame, "CONFIDENCE": conf, "IMAGE_PATH": img_path}
        for col in angle_cols:
            record[col] = angles.get(col, np.nan)
        records.append(record)

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False, float_format="%.2f")

    print(f"✅ Saved best 5 pose angles with image paths to: {out_csv}")
    return out_csv



# ==========================================================
# PIPELINE RUNNER
# ==========================================================

def run_full_pipeline(video_path: str) -> Dict[str, str]:
    """
    Run the full pipeline on the given video path.

    Returns:
        dict with:
            - masked_video_path
            - best_pose_frames_summary_csv
            - best_pose_angles_summary_csv
    """
    global INPUT_VIDEO
    INPUT_VIDEO = video_path

    # 1) YOLO compositor
    yolo_out = main_yolo_keep_one_object()
    masked_video_path = yolo_out["masked_video_path"]

    # 2) Frames -> crop
    _ = step2_extract_frames(masked_video_path)
    step2_auto_crop_all()

    # 3) Landmarks -> Kalman -> skeleton -> angles
    xy_csv = step2_extract_landmarks_xy()
    kalman_csv = step2_apply_kalman_filter(xy_csv)
    skeleton_dir = step2_draw_skeletons_from_kalman(kalman_csv)
    angles_csv = step2_calculate_angles_from_kalman(kalman_csv)

    # 4) Classification
    classified_csv = classify_from_kalman_angles(angles_csv)
    print("\n✅ Classification complete! Results saved in:", classified_csv)

    # 5) Best frame per pose
    best_frames_summary_csv = export_best_pose_frames(
        classified_csv, skeleton_folder=skeleton_dir
    )

    # 6) Best angles summary for those frames
    best_angles_summary_csv = visualize_best_pose_angles(
        best_summary_csv=best_frames_summary_csv,
        kalman_angles_csv=angles_csv,
        out_csv=os.path.join(BASE_DIR, "best_pose_angles_summary.csv"),
    )

    print("\n[PIPELINE DONE] All stages completed successfully.")
    return {
        "masked_video_path": masked_video_path,
        "best_pose_frames_summary_csv": best_frames_summary_csv,
        "best_pose_angles_summary_csv": best_angles_summary_csv,
    }
