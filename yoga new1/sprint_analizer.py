"""
Keep-One-Object Video Compositor + Pose Pipeline (Unified Single File)
----------------------------------------------------------------------
This file merges your three original scripts into one end-to-end pipeline,
while preserving your original functions, variables, thresholds, and folders.

ORDER OF EXECUTION:
  1) YOLOv8 Keep-One-Object Video Compositor (with ROI/class select)
  2) Extract frames from masked video -> Auto-crop -> MediaPipe overlays -> Skeleton-only PNGs
  3) Export pose angles CSV -> Rule-based classification -> Top-K per pose -> Annotated images
     -> (Optional) CSV printer & image viewer

You can still run each section's functions independently if you want.
"""

# ===========================
# ========== IMPORTS =========
# ===========================
import os
import cv2
import time
import shutil
import glob
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List

# MediaPipe is imported lazily in the sections that need it (as in your scripts).


# =============================================================================
# ========== SECTION 1: Keep-One-Object Video Compositor (YOUR ORIGINAL) ======
# =============================================================================

# ========= USER SETTINGS (EDIT HERE) =========================================
INPUT_VIDEO = r"E:\Evoq\Sprinting video\useful\IMG_0772.MOV"  # <--- your path

# Choose ONE selection method:
SELECTION_METHOD = "roi"    # "roi" or "class"
INIT_FRAME_INDEX = 95       # frame index to show for ROI and to start tracking

# If SELECTION_METHOD == "class":
CLASS_LABEL = "person"      # e.g., "person", "dog", "car", ...

# Camera mode:
CAMERA_MODE = "auto"        # "auto" | "static" | "moving"
USE_BG_SUB_FOR_STATIC = True

# Model selection:
MODEL_SIZE = "yolov8s-seg"  # "yolov8n-seg" (faster), "yolov8s-seg", ...
MIN_CONF = 0.25
NMS_IOU = 0.5

# Tracking / Re-ID:
IOU_SWITCH_THRESH = 0.25     # 0.15–0.30 typical
MAX_MISSES = 30
REID_MIN_OVERLAP = 0.05

# Warm-up acceptance for first frames after start
WARMUP_FRAMES = 10
WARMUP_IOU_THR = 0.12
WARMUP_BEST_SCORE = 0.10
STEADY_BEST_SCORE = 0.15

# Mask smoothing & post-process:
MASK_SMOOTH_ALPHA = 0.35     # EMA smoothing on OUTPUT mask (silhouette or box)
MORPH_OPEN = 3
MORPH_CLOSE = 5
FEATHER_PX = 6

# NEW — Keep the entire rectangular box instead of just the silhouette:
KEEP_ENTIRE_BOX = True              # set True to keep the whole box region
BOX_MODE = "fixed_from_init"        # "fixed_from_init" | "tight_to_detection"
BOX_PAD_PX = 20                     # padding around detection for box modes
BOX_CENTER_SMOOTH_ALPHA = 0.5       # smoothing for box center (0..1)
BOX_SIZE_SMOOTH_ALPHA = 0.4         # smoothing for box W/H when tight_to_detection

# Export (paths are bases; codec may change extension for compatibility)
WRITE_MASKED = "output_masked4.mp4"
WRITE_MASK = "output_mask.mp4"
SAVE_PNG_SEQUENCE = False
PNG_DIR = "frames_rgba"

# Preview window:
PREVIEW = True
PREVIEW_INPAINT_COMPARE = False
PAUSE_ON_START = False

# ROI display size (fit to screen)
ROI_WINDOW_MAX = (1280, 720)
PRINT_EVERY = 30
# =============================================================================


def ensure_dir(path: str):
    if os.path.isdir(path):
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
    picks = [min(i*step, length-1) for i in range(sample)]
    frames = []
    for p in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(f, (320, int(320*f.shape[0]/f.shape[1]))))
    for i in range(len(frames)-1):
        diff = cv2.absdiff(frames[i], frames[i+1])
        total += float(np.mean(diff))
        count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
    if count == 0:
        return True
    return (total/count) < thresh


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
        clss  = boxes.cls.detach().cpu().numpy().astype(int)
    except Exception:
        masks_np = np.array(masks)
        xyxy = np.array(boxes.xyxy).astype(int)
        confs = np.array(boxes.conf)
        clss  = np.array(boxes.cls).astype(int)
    for i in range(masks_np.shape[0]):
        if confs[i] < min_conf:
            continue
        m = masks_np[i]
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (m > 0.5).astype(np.uint8)
        x1, y1, x2, y2 = xyxy[i]
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H, y2))
        out.append({"mask": m, "bbox": (x1, y1, x2, y2), "cls": clss[i], "conf": float(confs[i])})
    return out


def iou_box(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    if union <= 0:
        return 0.0
    return inter/union


def overlap_ratio(a_mask: np.ndarray, b_mask: np.ndarray) -> float:
    inter = np.logical_and(a_mask>0, b_mask>0).sum()
    denom = max(1, (a_mask>0).sum())
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
        blurred = cv2.GaussianBlur((m*255).astype(np.uint8), (0,0), sigmaX=max(1, feather_px/2))
        return blurred
    return (m*255).astype(np.uint8)


def inpaint_hole(bg: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    hole = (mask_u8 == 0).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
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


def rect_from_center_size(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int,int,int,int]:
    x1 = int(round(cx - w/2))
    y1 = int(round(cy - h/2))
    x2 = x1 + int(round(w))
    y2 = y1 + int(round(h))
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2)); y2 = max(0, min(H,   y2))
    if x2 <= x1: x2 = min(W, x1+1)
    if y2 <= y1: y2 = min(H, y1+1)
    return x1, y1, x2, y2


def mask_from_rect(rect: Tuple[int,int,int,int], H: int, W: int) -> np.ndarray:
    x1, y1, x2, y2 = rect
    m = np.zeros((H, W), dtype=np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def main_yolo_keep_one_object() -> Dict[str, Any]:
    """
    Runs your original Keep-One-Object compositor.
    Returns a dict with the actual output paths for masked video and mask video.
    """
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_VIDEO}")
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
                          imgsz=max(640, ((max(W,H)+31)//32)*32))

    bg_sub = None
    if cam_mode == "static" and USE_BG_SUB_FOR_STATIC:
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
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
        raise RuntimeError("No instance found on the initialization frame. "
                           "Try a different frame index, lower MIN_CONF, or a larger model.")

    class_id_to_name = getattr(model.model, "names", getattr(model, "names", {}))
    name_to_class_id = {v: k for k, v in class_id_to_name.items()}

    target_cls_id = None
    target_mask = None
    target_bbox = None
    target_id_text = None

    init_roi_box = None
    box_cx_f = None
    box_cy_f = None
    box_w_f  = None
    box_h_f  = None

    if SELECTION_METHOD == "class":
        if CLASS_LABEL not in name_to_class_id:
            raise RuntimeError(f"CLASS_LABEL '{CLASS_LABEL}' not in model classes. "
                               f"Available: {sorted(set(class_id_to_name.values()))[:20]} ...")
        wanted_id = name_to_class_id[CLASS_LABEL]
        cands = [ins for ins in init_instances if ins["cls"] == wanted_id]
        if not cands:
            raise RuntimeError(f"No '{CLASS_LABEL}' found on frame {base_idx}. Try another frame or label.")
        pick = max(cands, key=lambda d: d["conf"])
        target_cls_id = wanted_id
        target_mask = pick["mask"]
        target_bbox = pick["bbox"]
        target_id_text = f"{CLASS_LABEL} (init conf={pick['conf']:.2f})"
        print(f"[INFO] Selected class: {CLASS_LABEL} (id={target_cls_id}), conf={pick['conf']:.2f}")

        if KEEP_ENTIRE_BOX:
            x1,y1,x2,y2 = target_bbox
            w0 = max(1, (x2-x1) + 2*BOX_PAD_PX)
            h0 = max(1, (y2-y1) + 2*BOX_PAD_PX)
            cx0 = (x1+x2)/2
            cy0 = (y1+y2)/2
            if BOX_MODE == "fixed_from_init":
                box_w_f, box_h_f = float(w0), float(h0)
            else:
                box_w_f, box_h_f = float(w0), float(h0)
            box_cx_f, box_cy_f = float(cx0), float(cy0)

    elif SELECTION_METHOD == "roi":
        src_h, src_w = init_frame.shape[:2]
        max_w, max_h = ROI_WINDOW_MAX
        scale_w = max_w / src_w
        scale_h = max_h / src_h
        scale = min(1.0, min(scale_w, scale_h))  # never upscale
        win_w = int(src_w * scale)
        win_h = int(src_h * scale)
        disp = cv2.resize(init_frame, (win_w, win_h)) if scale != 1.0 else init_frame.copy()
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", win_w, win_h)
        cv2.putText(disp, "Draw ROI around the object to KEEP. Press ENTER.",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        roi = cv2.selectROI("Select ROI", disp, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        (rx, ry, rw, rh) = roi
        if rw <= 0 or rh <= 0:
            raise RuntimeError("ROI selection cancelled or invalid.")
        rx = int(rx / scale); ry = int(ry / scale)
        rw = int(rw / scale); rh = int(rh / scale)
        init_roi_box = (rx, ry, rx+rw, ry+rh)

        best = None
        best_iou = -1.0
        for ins in init_instances:
            iou = iou_box(ins["bbox"], init_roi_box)
            if iou > best_iou:
                best_iou = iou
                best = ins
        if best is None or best_iou <= 0.0:
            raise RuntimeError("No instance overlaps with the chosen ROI. Try again or use CLASS selection.")
        target_cls_id = best["cls"]
        target_mask = best["mask"]
        target_bbox = best["bbox"]
        target_id_text = f"{class_id_to_name.get(target_cls_id, str(target_cls_id))} (IoU with ROI={best_iou:.2f})"
        print(f"[INFO] Selected by ROI -> class={class_id_to_name.get(target_cls_id,'?')} id={target_cls_id}, IoU={best_iou:.3f}")

        if KEEP_ENTIRE_BOX:
            rx1, ry1, rx2, ry2 = init_roi_box
            w0 = rx2 - rx1
            h0 = ry2 - ry1
            cx0 = (rx1 + rx2) / 2.0
            cy0 = (ry1 + ry2) / 2.0
            if BOX_MODE == "fixed_from_init":
                box_w_f, box_h_f = float(w0), float(h0)  # fixed from ROI
            else:
                x1,y1,x2,y2 = target_bbox
                box_w_f = float(max(1, (x2-x1) + 2*BOX_PAD_PX))
                box_h_f = float(max(1, (y2-y1) + 2*BOX_PAD_PX))
            box_cx_f, box_cy_f = float(cx0), float(cy0)
    else:
        raise RuntimeError("SELECTION_METHOD must be 'roi' or 'class'.")

    print(f"[INFO] Will start tracking from frame {base_idx}")

    prev_mask_f = None
    last_assoc_mask = (target_mask>0).astype(np.uint8)
    last_bbox = target_bbox
    misses = 0

    t0 = time.time()
    frame_counter = 0

    if PREVIEW and PAUSE_ON_START:
        print("[INFO] Press any key in preview window to start...")
        cv2.imshow("preview", init_frame)
        cv2.waitKey(0)

    vw_masked, masked_path, FPS_OUT = make_video_writer(WRITE_MASKED, (W, H), FPS, is_color=True)
    vw_mask,   mask_path,   _      = make_video_writer(WRITE_MASK,   (W, H), FPS, is_color=False)

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
            assoc_cur_mask = (picked["mask"]>0).astype(np.uint8)
            last_assoc_mask = assoc_cur_mask
            last_bbox = picked["bbox"]
            misses = 0

            if KEEP_ENTIRE_BOX:
                px1,py1,px2,py2 = picked["bbox"]
                cur_cx = (px1 + px2)/2.0
                cur_cy = (py1 + py2)/2.0
                if box_cx_f is None:
                    box_cx_f, box_cy_f = float(cur_cx), float(cur_cy)
                else:
                    box_cx_f = (1.0 - BOX_CENTER_SMOOTH_ALPHA)*box_cx_f + BOX_CENTER_SMOOTH_ALPHA*cur_cx
                    box_cy_f = (1.0 - BOX_CENTER_SMOOTH_ALPHA)*box_cy_f + BOX_CENTER_SMOOTH_ALPHA*cur_cy

                if BOX_MODE == "tight_to_detection":
                    cur_w = max(1, (px2-px1) + 2*BOX_PAD_PX)
                    cur_h = max(1, (py2-py1) + 2*BOX_PAD_PX)
                    if box_w_f is None:
                        box_w_f, box_h_f = float(cur_w), float(cur_h)
                    else:
                        box_w_f = (1.0 - BOX_SIZE_SMOOTH_ALPHA)*box_w_f + BOX_SIZE_SMOOTH_ALPHA*cur_w
                        box_h_f = (1.0 - BOX_SIZE_SMOOTH_ALPHA)*box_h_f + BOX_SIZE_SMOOTH_ALPHA*cur_h
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
                x1,y1,x2,y2 = last_bbox
                box_cx_f = (x1+x2)/2.0; box_cy_f = (y1+y2)/2.0
                if BOX_MODE == "tight_to_detection":
                    box_w_f = max(1, (x2-x1) + 2*BOX_PAD_PX)
                    box_h_f = max(1, (y2-y1) + 2*BOX_PAD_PX)
                else:
                    if init_roi_box is not None:
                        rx1,ry1,rx2,ry2 = init_roi_box
                        box_w_f = float(rx2-rx1); box_h_f = float(ry2-ry1)
                    else:
                        box_w_f = max(1, (x2-x1) + 2*BOX_PAD_PX)
                        box_h_f = max(1, (y2-y1) + 2*BOX_PAD_PX)

            rect = rect_from_center_size(box_cx_f, box_cy_f, box_w_f, box_h_f, W, H)
            output_cur_mask = mask_from_rect(rect, H, W)
        else:
            output_cur_mask = last_assoc_mask.copy()

        prev_mask_f = ema_mask(prev_mask_f, output_cur_mask, alpha=MASK_SMOOTH_ALPHA)
        smoothed_u8 = (np.clip(prev_mask_f, 0, 1)*255).astype(np.uint8)

        post_mask_u8 = postprocess_mask((smoothed_u8 > 127).astype(np.uint8),
                                        FEATHER_PX, MORPH_OPEN, MORPH_CLOSE)

        if post_mask_u8.shape[:2] != frame.shape[:2]:
            post_mask_u8 = cv2.resize(post_mask_u8, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

        alpha = post_mask_u8.astype(np.float32) / 255.0
        alpha3 = np.dstack([alpha, alpha, alpha])
        out_masked = (frame.astype(np.float32) * alpha3).astype(np.uint8)

        vw_masked.write(out_masked)
        vw_mask.write(post_mask_u8)

        if SAVE_PNG_SEQUENCE:
            if not os.path.isdir(PNG_DIR):
                ensure_dir(PNG_DIR)
            b, g, r = cv2.split(frame)
            a = post_mask_u8
            rgba = cv2.merge([b, g, r, a])
            cv2.imwrite(os.path.join(PNG_DIR, f"frame_{idx:06d}.png"), rgba)

        if PREVIEW:
            vis = out_masked.copy()
            hud = f"Frame {idx}/{N_FRAMES-1}  misses={misses}  target={class_id_to_name.get(target_cls_id,'?')}"
            cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            if PREVIEW_INPAINT_COMPARE:
                hole_filled = inpaint_hole(frame, post_mask_u8)
                side = np.hstack([vis, hole_filled])
                cv2.imshow("preview", side)
            else:
                cv2.imshow("preview", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[INFO] ESC pressed. Stopping.")
                break
            if key == ord(' '):
                cv2.waitKey(0)

        frame_counter += 1
        if frame_counter % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            fps = frame_counter / max(1e-6, elapsed)
            print(f"[INFO] Processed {frame_counter} frames @ {fps:.2f} FPS")

    cap.release()
    vw_masked.release()
    vw_mask.release()
    if PREVIEW:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    total_time = time.time() - t0
    if frame_counter > 0:
        print(f"[DONE] Wrote '{masked_path}' and '{mask_path}'.")
        print(f"[STATS] Frames={frame_counter}, Avg FPS={frame_counter/max(1e-6,total_time):.2f}")
    else:
        print("[DONE] No frames processed.")

    return {"masked_video_path": masked_path, "mask_video_path": mask_path}


# =============================================================================
# ========== SECTION 2: Frames Extract + Auto-Crop + MediaPipe ================
# =============================================================================

# Your original folder names preserved:
RAW_DIR = "exported_frames_raw"
CROPPED_DIR = "exported_frames_cropped2"
OUT_DIR_POSE_OVERLAY = "exported frames with mp2"
SKEL_DIR = "exported frames mp sketch only"
OUT_CSV = "pose_angles_by_frame2.csv"

# Drawing params for skeleton-only:
VIS_THRESH = 0.30
LINE_SCALE = 150
DOT_SCALE  = 250
SKEL_RGB   = (255, 255, 255)  # white
SWAP_AXES  = False  # for the angles export part


def step2_extract_frames(video_path: str) -> Dict[str, Any]:
    print("\n[STEP 2.1] Extract frames from video ->", video_path)
    os.makedirs(RAW_DIR, exist_ok=True)
    for f in Path(RAW_DIR).glob("*.png"):
        try: f.unlink()
        except Exception as e: print("Warn:", e)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        cv2.imwrite(str(Path(RAW_DIR)/f"frame_{frame_idx:06d}.png"), frame)

    cap.release()

    print(f"Video path       : {video_path}")
    print(f"Resolution (wxh) : {w} x {h}")
    print(f"FPS (metadata)   : {fps:.3f}")
    print(f"Frames extracted : {frame_idx}")
    return {"fps": fps, "w": w, "h": h, "frames": frame_idx}


def auto_crop(img, pad_frac=0.02, min_area_ratio=0.01):
    """
    Find the largest bright/non-black region and crop to its bounding box.
    pad_frac: extra margin around the box (fraction of max(w,h)).
    min_area_ratio: ignore tiny blobs (< this ratio of full frame).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, (0,0,w,h)

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < (w*h*min_area_ratio):
        return img, (0,0,w,h)

    x, y, bw, bh = cv2.boundingRect(c)
    pad = int(round(max(bw, bh) * pad_frac))

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)

    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def step2_auto_crop_all():
    print("\n[STEP 2.2] Auto-crop all frames")
    os.makedirs(CROPPED_DIR, exist_ok=True)
    for f in Path(CROPPED_DIR).glob("*.png"):
        try: f.unlink()
        except Exception as e: print("Warn:", e)

    raw_files = sorted(Path(RAW_DIR).glob("*.png"))
    if not raw_files:
        raise RuntimeError(f"No frames found in {RAW_DIR}. Run extraction first.")

    saved = 0
    for i, fp in enumerate(raw_files, start=1):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        cropped, bbox = auto_crop(img, pad_frac=0.02)
        cv2.imwrite(str(Path(CROPPED_DIR)/f"crop_{i:06d}.png"), cropped)
        saved += 1

    print(f"Cropped {saved} frames → '{CROPPED_DIR}'")


def step2_pose_overlay_previews():
    print("\n[STEP 2.3] MediaPipe Pose overlay on cropped frames (preview/save)")
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError("MediaPipe not found. Please run: pip install mediapipe") from e

    mp_pose  = mp.solutions.pose
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    files = sorted(Path(CROPPED_DIR).glob("*.png"))
    if not files:
        raise RuntimeError(f"No cropped frames found in {CROPPED_DIR}. Run crop step first.")

    os.makedirs(OUT_DIR_POSE_OVERLAY, exist_ok=True)
    for f in Path(OUT_DIR_POSE_OVERLAY).glob("*.png"):
        try: f.unlink()
        except Exception as e: print("Warn:", e)

    saved = 0
    for idx, fp in enumerate(files, start=1):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
            )
        cv2.imwrite(str(Path(OUT_DIR_POSE_OVERLAY)/f"mp_{idx:06d}.png"), img)
        saved += 1

    pose.close()
    print(f"Saved {saved} processed frames → '{OUT_DIR_POSE_OVERLAY}'")


def step2_skeleton_only_pngs():
    print("\n[STEP 2.4] Skeleton-only transparent PNGs (from cropped frames)")
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError("MediaPipe not found. Please run: pip install mediapipe") from e

    os.makedirs(SKEL_DIR, exist_ok=True)
    for f in Path(SKEL_DIR).glob("*.png"):
        try: f.unlink()
        except Exception as e: print("Warn:", e)

    mp_pose = mp.solutions.pose
    POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    files = sorted(Path(CROPPED_DIR).glob("*.png"))
    if not files:
        raise RuntimeError(f"No cropped frames found in '{CROPPED_DIR}'. Run crop step first.")

    saved = 0
    for i, fp in enumerate(files, start=1):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]

        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        alpha = np.zeros((h, w), dtype=np.uint8)

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            thick = max(2, int(round(min(h, w) / LINE_SCALE)))
            rad   = max(2, int(round(min(h, w) / DOT_SCALE)))

            def lm_px(idx):
                lm = lms[idx]
                if lm.visibility is not None and lm.visibility < VIS_THRESH:
                    return None
                x = int(round(lm.x * w))
                y = int(round(lm.y * h))
                if x < 0 or y < 0 or x >= w or y >= h:
                    return None
                return (x, y)

            for a, b in POSE_CONNECTIONS:
                pa = lm_px(a); pb = lm_px(b)
                if pa is not None and pb is not None:
                    cv2.line(alpha, pa, pb, color=255, thickness=thick, lineType=cv2.LINE_AA)

            for idx in range(len(lms)):
                p = lm_px(idx)
                if p is not None:
                    cv2.circle(alpha, p, rad, color=255, thickness=-1, lineType=cv2.LINE_AA)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0:3][alpha > 0] = SKEL_RGB
        rgba[..., 3] = alpha

        out_path = Path(SKEL_DIR) / f"skeleton_{i:06d}.png"
        cv2.imwrite(str(out_path), rgba)
        saved += 1

    pose.close()
    print(f"Saved {saved} skeleton-only PNGs → '{SKEL_DIR}'")


def step2_export_angles_csv():
    print("\n[STEP 2.5] Export angles CSV (pelvis-normalized; optional axis swap)")
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError("Please install MediaPipe: pip install mediapipe") from e

    import pandas as pd

    PL = mp.solutions.pose.PoseLandmark

    ANGLES = {
        "LEFT_SHOULDER" : (PL.LEFT_HIP, PL.LEFT_SHOULDER, PL.LEFT_ELBOW),
        "RIGHT_SHOULDER": (PL.RIGHT_HIP, PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW),
        "LEFT_ELBOW"    : (PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST),
        "RIGHT_ELBOW"   : (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST),
        "LEFT_WRIST"    : (PL.LEFT_ELBOW, PL.LEFT_WRIST, PL.LEFT_INDEX),
        "RIGHT_WRIST"   : (PL.RIGHT_ELBOW, PL.RIGHT_WRIST, PL.RIGHT_INDEX),
        "LEFT_HIP"      : (PL.LEFT_SHOULDER, PL.LEFT_HIP, PL.LEFT_KNEE),
        "RIGHT_HIP"     : (PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE),
        "LEFT_KNEE"     : (PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE),
        "RIGHT_KNEE"    : (PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE),
        "LEFT_ANKLE"    : (PL.LEFT_KNEE, PL.LEFT_ANKLE, PL.LEFT_FOOT_INDEX),
        "RIGHT_ANKLE"   : (PL.RIGHT_KNEE, PL.RIGHT_ANKLE, PL.RIGHT_FOOT_INDEX),
        "LEFT_HEEL_ANGLE"  : (PL.LEFT_ANKLE, PL.LEFT_HEEL, PL.LEFT_FOOT_INDEX),
        "RIGHT_HEEL_ANGLE" : (PL.RIGHT_ANKLE, PL.RIGHT_HEEL, PL.RIGHT_FOOT_INDEX),
        "LEFT_TOE_ANGLE"   : (PL.LEFT_HEEL, PL.LEFT_FOOT_INDEX, PL.LEFT_ANKLE),
        "RIGHT_TOE_ANGLE"  : (PL.RIGHT_HEEL, PL.RIGHT_FOOT_INDEX, PL.RIGHT_ANKLE),
    }

    def angle_at_B(pa, pb, pc):
        if pa is None or pb is None or pc is None:
            return np.nan
        ba = pa - pb
        bc = pc - pb
        if np.allclose(ba, 0) or np.allclose(bc, 0):
            return np.nan
        cross = ba[0]*bc[1] - ba[1]*bc[0]
        dot   = ba[0]*bc[0] + ba[1]*bc[1]
        return math.degrees(math.atan2(abs(cross), dot))

    def lm_xy(lms, idx, w, h, vis_thresh=0.30):
        lm = lms[idx]
        if (lm.visibility is not None) and (lm.visibility < vis_thresh):
            return None
        x = int(round(lm.x * w))
        y = int(round(lm.y * h))
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return np.array([x, y], dtype=np.float32)

    def apply_axis_swap(p):
        if p is None:
            return None
        return np.array([p[1], p[0]], dtype=np.float32) if SWAP_AXES else p

    files = sorted(Path(CROPPED_DIR).glob("*.png"))
    if not files:
        raise RuntimeError(f"No frames found in '{CROPPED_DIR}'. Make sure you ran the crop step.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )

    angle_names = list(ANGLES.keys())
    records = []

    for fp in files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        row = {"FRAME_NAME": fp.name}
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            lh = lm_xy(lms, mp_pose.PoseLandmark.LEFT_HIP,  w, h, VIS_THRESH)
            rh = lm_xy(lms, mp_pose.PoseLandmark.RIGHT_HIP, w, h, VIS_THRESH)
            if lh is not None and rh is not None:
                pelvis = (lh + rh) / 2.0
            elif lh is not None:
                pelvis = lh
            elif rh is not None:
                pelvis = rh
            else:
                pelvis = None

            def lm_xy_norm(idx):
                p = lm_xy(lms, idx, w, h, VIS_THRESH)
                if p is None:
                    return None
                if pelvis is not None:
                    p = p - pelvis
                return apply_axis_swap(p)

            for name, (A, B, C) in ANGLES.items():
                pa = lm_xy_norm(A)
                pb = lm_xy_norm(B)
                pc = lm_xy_norm(C)
                row[name] = angle_at_B(pa, pb, pc)
        else:
            for name in angle_names:
                row[name] = np.nan

        records.append(row)

    pose.close()

    import pandas as pd
    df = pd.DataFrame.from_records(records)
    df = df.set_index("FRAME_NAME")
    df = df[angle_names]
    df.to_csv(OUT_CSV, float_format="%.3f")

    print(f"Done. Wrote {OUT_CSV}")
    print(f"Rows (frames): {df.shape[0]}, Columns (angles): {df.shape[1]}")
    return OUT_CSV


# =============================================================================
# ========== SECTION 3: Rule-based Classification + Top-K + CSV/Viewer ========
# =============================================================================

# Your original classifier & utilities preserved:

try:
    import mediapipe as mp
except Exception:
    mp = None  # will import inside functions as needed

def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    ab = [a[0] - b[0], a[1] - b[1]]
    cb = [c[0] - b[0], c[1] - b[1]]

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)

    if mag_ab == 0 or mag_cb == 0:
        return 0.0

    cosang = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    angle = math.degrees(math.acos(cosang))
    return angle


def compute_joint_angles(lm, mp_pose_=None):
    if mp_pose_ is None:
        import mediapipe as mp
        mp_pose_ = mp.solutions.pose

    angles = {}
    angles["Right Knee"] = calculate_angle(lm[mp_pose_.PoseLandmark.RIGHT_HIP],
                                           lm[mp_pose_.PoseLandmark.RIGHT_KNEE],
                                           lm[mp_pose_.PoseLandmark.RIGHT_ANKLE])

    angles["Left Knee"]  = calculate_angle(lm[mp_pose_.PoseLandmark.LEFT_HIP],
                                           lm[mp_pose_.PoseLandmark.LEFT_KNEE],
                                           lm[mp_pose_.PoseLandmark.LEFT_ANKLE])

    angles["Right Hip"]  = calculate_angle(lm[mp_pose_.PoseLandmark.RIGHT_SHOULDER],
                                           lm[mp_pose_.PoseLandmark.RIGHT_HIP],
                                           lm[mp_pose_.PoseLandmark.RIGHT_KNEE])

    angles["Left Hip"]   = calculate_angle(lm[mp_pose_.PoseLandmark.LEFT_SHOULDER],
                                           lm[mp_pose_.PoseLandmark.LEFT_HIP],
                                           lm[mp_pose_.PoseLandmark.LEFT_KNEE])

    angles["Right Elbow"]= calculate_angle(lm[mp_pose_.PoseLandmark.RIGHT_SHOULDER],
                                           lm[mp_pose_.PoseLandmark.RIGHT_ELBOW],
                                           lm[mp_pose_.PoseLandmark.RIGHT_WRIST])

    angles["Left Elbow"] = calculate_angle(lm[mp_pose_.PoseLandmark.LEFT_SHOULDER],
                                           lm[mp_pose_.PoseLandmark.LEFT_ELBOW],
                                           lm[mp_pose_.PoseLandmark.LEFT_WRIST])

    angles["Right Shoulder"] = calculate_angle(lm[mp_pose_.PoseLandmark.RIGHT_ELBOW],
                                               lm[mp_pose_.PoseLandmark.RIGHT_SHOULDER],
                                               lm[mp_pose_.PoseLandmark.RIGHT_HIP])

    angles["Left Shoulder"]  = calculate_angle(lm[mp_pose_.PoseLandmark.LEFT_ELBOW],
                                               lm[mp_pose_.PoseLandmark.LEFT_SHOULDER],
                                               lm[mp_pose_.PoseLandmark.LEFT_HIP])

    angles["Right Ankle (Heel-to-Toe)"] = calculate_angle(lm[mp_pose_.PoseLandmark.RIGHT_KNEE],
                                                          lm[mp_pose_.PoseLandmark.RIGHT_ANKLE],
                                                          lm[mp_pose_.PoseLandmark.RIGHT_FOOT_INDEX])

    angles["Left Ankle (Heel-to-Toe)"]  = calculate_angle(lm[mp_pose_.PoseLandmark.LEFT_KNEE],
                                                          lm[mp_pose_.PoseLandmark.LEFT_ANKLE],
                                                          lm[mp_pose_.PoseLandmark.LEFT_FOOT_INDEX])
    return angles


POSE_LIST = ["Toe Off", "MVP", "Strike", "Touch Down", "Full Support"]

def classify_pose(angles, min_hits=4):
    rk, lk = angles["Right Knee"],  angles["Left Knee"]
    rh, lh = angles["Right Hip"],   angles["Left Hip"]
    re, le = angles["Right Elbow"], angles["Left Elbow"]
    rs, ls = angles["Right Shoulder"], angles["Left Shoulder"]
    ra, la = angles["Right Ankle (Heel-to-Toe)"], angles["Left Ankle (Heel-to-Toe)"]

    pose_scores = {k: 0 for k in ["Toe Off","MVP","Strike","Touch Down","Full Support"]}

    rules = {
        "Toe Off": [
            75 < rk < 85,    160 < lk < 175,  95 < rh < 115,  160 < lh < 175,
            95 < re < 140,   80 < le < 95,   55< rs < 75,  40 < ls < 65,
            95 < ra < 135,   130 < la < 145
        ],
        "MVP": [
            120 < rk < 140,  115 < lk < 127, 150 < rh < 175,  120 < lh < 145,
            47 < re < 67,    130 < le < 150, 55 < rs < 70,    65 < ls < 85,
            120 < ra < 140,  125 < la < 150
        ],
        "Strike": [
            55 < rk < 75,    150 < lk < 165, 170 < rh < 180,  130 < lh < 140,
            100 < re < 115,  115 < le < 130, 10 < rs < 25,    30 < ls < 45,
            140 < ra < 170,   90 < la < 110
        ],
        "Touch Down": [
            45 < rk < 60,    155 < lk < 170, 150 < rh < 170,  145 < lh < 160,
            130 < re < 150,  105 < le < 120, 0 < rs < 10,     10 < ls < 20,
            130 < ra < 160,   85 < la < 105
        ],
        "Full Support": [
            15 < rk < 30,    135 < lk < 160, 130 < rh < 145,  150 < lh < 165,
            155 < re < 175,  100 < le < 130, 0 < rs < 10,     5 < ls < 15,
            95 < ra < 115,   120 < la < 150
        ],
    }

    for pose_name, conditions in rules.items():
        pose_scores[pose_name] = int(np.sum(conditions))

    best_pose = max(pose_scores, key=pose_scores.get)
    best_score = pose_scores[best_pose]
    max_possible = len(rules[best_pose])

    if best_score < min_hits:
        return "Unknown Pose", 0.0

    confidence = round((best_score / max_possible) * 100.0, 1)
    return best_pose, confidence


def _annotate_and_save(image, results, label, confidence, out_path):
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    img = image.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    cv2.putText(img, f"{label} ({confidence}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 255, 0) if label != 'Unknown Pose' else (0, 0, 255), 3)
    cv2.imwrite(out_path, img)

# Avoid repeating
import hashlib

def _file_md5(path, chunk_size=1<<20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
#
def analyze_folder_topk(folder_path,
                        top_k=2,
                        min_hits=4,
                        save_best=True,
                        out_dir=None,
                        angles_csv_path=None,
                        dedupe_across_poses=False):
    """
    Scans a folder of frames, classifies each frame, and picks the top_k frames per pose.
    Saves annotated copies to `out_dir` and writes a CSV with angles (from live detection here).
    """
    if out_dir is None:
        out_dir = os.path.join(folder_path, "best_pose_frames_topk")
    os.makedirs(out_dir, exist_ok=True)
    if angles_csv_path is None:
        angles_csv_path = os.path.join(out_dir, "selected_pose_angles.csv")

    patterns = ['*.png','*.jpg','*.jpeg','*.bmp','*.JPG','*.PNG','*.JPEG']
    frame_paths = []
    for p in patterns:
        frame_paths.extend(glob.glob(os.path.join(folder_path, p)))
    frame_paths = sorted(frame_paths)

    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=1)

    per_pose = defaultdict(list)
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            continue
        angles = compute_joint_angles(res.pose_landmarks.landmark, mp_pose_=mp_pose)
        label, conf = classify_pose(angles, min_hits=min_hits)
        if label in POSE_LIST:
            per_pose[label].append({
                "path": fp,
                "confidence": float(conf),
                "angles": angles
            })
# Avoiding repeat
    selected = {}
    used_paths = set()
    # If dedupe_across_poses=True, also dedupe by content across all poses:
    used_hashes_global = set() if dedupe_across_poses else None

    for pose_name in POSE_LIST:
        # sort by confidence, high to low
        candidates = sorted(per_pose.get(pose_name, []),
                            key=lambda d: d["confidence"], reverse=True)

        # --- Strong dedupe step: remove duplicate files by MD5 content hash ---
        unique_candidates = []
        seen_hashes = set()
        for d in candidates:
            try:
                h = _file_md5(d["path"])
            except Exception:
                # if hashing fails, fall back to path-based dedupe
                h = os.path.normcase(os.path.normpath(os.path.abspath(d["path"])))

            # per-pose dedupe
            if h in seen_hashes:
                continue

            # optional cross-pose dedupe (only if flag set)
            if used_hashes_global is not None and h in used_hashes_global:
                continue

            seen_hashes.add(h)
            if used_hashes_global is not None:
                used_hashes_global.add(h)

            d = dict(d)
            d["_hash"] = h
            unique_candidates.append(d)

        # now select top_k without duplicates
        if dedupe_across_poses:
            top = []
            for d in unique_candidates:
                # keep existing “across poses” path-level guard too
                if d["path"] not in used_paths:
                    top.append(d)
                    used_paths.add(d["path"])
                if len(top) == top_k:
                    break
        else:
            top = unique_candidates[:top_k]

        selected[pose_name] = top
#
        if save_best and top:
            subdir = os.path.join(out_dir, pose_name.replace(" ", "_"))
            os.makedirs(subdir, exist_ok=True)
            for rank, item in enumerate(top, 1):
                img = cv2.imread(item["path"])
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                base = os.path.basename(item["path"])
                out_path = os.path.join(
                    subdir,
                    f"{rank:02d}__{pose_name}__{item['confidence']:.1f}__{base}"
                )
                _annotate_and_save(img, res, pose_name, item["confidence"], out_path)

    print("\n=== Top picks per pose ===")
    for pose_name in POSE_LIST:
        top = selected.get(pose_name, [])
        if not top:
            print(f"{pose_name}: (none)")
        else:
            for i, it in enumerate(top, 1):
                print(f"{pose_name} [{i}]: {it['path']} ({it['confidence']}%)")

    headers = [
        "Pose", "Rank", "Confidence(%)", "Filename", "Path",
        "Right Knee", "Left Knee",
        "Right Hip", "Left Hip",
        "Right Elbow", "Left Elbow",
        "Right Shoulder", "Left Shoulder",
        "Right Ankle (Heel-to-Toe)", "Left Ankle (Heel-to-Toe)"
    ]
    with open(angles_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for pose_name in POSE_LIST:
            for rank, item in enumerate(selected.get(pose_name, []), 1):
                ang = item["angles"]
                row = [
                    pose_name,
                    rank,
                    f"{item['confidence']:.1f}",
                    os.path.basename(item["path"]),
                    item["path"],
                    f"{ang['Right Knee']:.2f}°",
                    f"{ang['Left Knee']:.2f}°",
                    f"{ang['Right Hip']:.2f}°",
                    f"{ang['Left Hip']:.2f}°",
                    f"{ang['Right Elbow']:.2f}°",
                    f"{ang['Left Elbow']:.2f}°",
                    f"{ang['Right Shoulder']:.2f}°",
                    f"{ang['Left Shoulder']:.2f}°",
                    f"{ang['Right Ankle (Heel-to-Toe)']:.2f}°",
                    f"{ang['Left Ankle (Heel-to-Toe)']:.2f}°",
                ]
                w.writerow(row)

    print(f"\n✅ Angles CSV written to: {angles_csv_path}")
    pose.close()
    return selected


ANGLE_COLS = [
    "Right Knee", "Left Knee",
    "Right Hip", "Left Hip",
    "Right Elbow", "Left Elbow",
    "Right Shoulder", "Left Shoulder",
    "Right Ankle (Heel-to-Toe)", "Left Ankle (Heel-to-Toe)"
]

def _put_text_line(img, text, org, font_scale=0.6):
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)

def print_and_show_csv(angles_csv_path, draw_landmarks=True, show_images=True, max_rows=None):
    if not os.path.exists(angles_csv_path):
        print(f"[ERROR] CSV not found: {angles_csv_path}")
        return

    with open(angles_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_rows is not None:
        rows = rows[:max_rows]

    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=1)

    for idx, row in enumerate(rows, 1):
        pose_name   = row.get("Pose", "")
        rank        = row.get("Rank", "")
        conf        = row.get("Confidence(%)", "")
        filename    = row.get("Filename", "")
        path        = row.get("Path", "")

        print(f"\n[{idx}] Pose: {pose_name}  |  Rank: {rank}  |  Confidence: {conf}%")
        print(f"     File: {filename}")
        print(f"     Path: {path}")
        print( "     Angles:")
        for col in ANGLE_COLS:
            val = row.get(col, "")
            print(f"       - {col}: {val}")

        if show_images:
            img = cv2.imread(path)
            if img is None:
                print("     [WARN] Could not open image, skipping display.")
                continue

            if draw_landmarks:
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            _put_text_line(img, f"{pose_name}  ({conf}%)", (10, 30), font_scale=0.8)

            plt.figure(figsize=(7, 7))
            plt.title(f"{pose_name}  (Rank {rank}, {conf}%)")
            plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

    pose.close()


# =============================================================================
# ========== PIPELINE RUNNER (calls sections in sequence) ======================
# =============================================================================

def run_full_pipeline():
    # 1) YOLO keep-one-object compositor
    yolo_out = main_yolo_keep_one_object()
    masked_video_path = yolo_out["masked_video_path"]
    print(f"\n[PIPELINE] Masked video ready: {masked_video_path}")

    # 2.1) Extract frames from masked video
    _ = step2_extract_frames(masked_video_path)

    # 2.2) Auto-crop frames
    step2_auto_crop_all()

    # 2.3) MediaPipe Pose overlay (saved images)
    step2_pose_overlay_previews()

    # 2.4) Skeleton-only transparent PNGs
    step2_skeleton_only_pngs()

    # 2.5) Export angles CSV
    angles_csv_from_step2 = step2_export_angles_csv()

    # 3) Classification + Best-K per pose (use CROPPED_DIR as in your script)
    #    Save to a new OUT_DIR with your original naming style
    OUT_DIR_FINAL = Path(r"E:\yoga new\final poses")
    OUT_DIR_FINAL.mkdir(parents=True, exist_ok=True)
    ANGLES_CSV_FINAL = OUT_DIR_FINAL / "final_poses_angles.csv"

    bestk = analyze_folder_topk(
        folder_path=str(Path(CROPPED_DIR)),
        top_k=2,
        min_hits=4,
        save_best=True,
        out_dir=str(OUT_DIR_FINAL),
        angles_csv_path=str(ANGLES_CSV_FINAL),
        dedupe_across_poses=False
    )

    # Optional: print & show CSV rows (images with overlays)
    # Comment out if you don't want pop-up windows
    print_and_show_csv(str(ANGLES_CSV_FINAL), draw_landmarks=True, show_images=True, max_rows=None)

    print("\n[PIPELINE DONE] All steps completed.")


# =============================================================================
# ========== MAIN =============================================================
# =============================================================================
if __name__ == "__main__":
    try:
        run_full_pipeline()
    except RuntimeError as e:
        print("\n[ERROR]", e)
        print("\nTroubleshooting tips:")
        print("- Check INPUT_VIDEO path and codecs.")
        print("- If 'No instance found on the initialization frame':")
        print("    * Try a different INIT_FRAME_INDEX")
        print("    * Lower MIN_CONF or choose a larger MODEL_SIZE")
        print("    * Use ROI selection instead of class (or vice versa)")
        print("- If tracking drifts or box jitters:")
        print("    * Lower IOU_SWITCH_THRESH to 0.15–0.20 (moving camera)")
        print("    * Increase BOX_CENTER_SMOOTH_ALPHA (e.g., 0.7) for steadier box motion")
        print("    * If box size pumps, increase BOX_SIZE_SMOOTH_ALPHA or switch to fixed_from_init")
        print("- Edge softness: increase FEATHER_PX or MORPH_CLOSE.")
