"""
Keep-One-Object Video Compositor (Strict Seek + Warm-up + Box Mode)
-------------------------------------------------------------------
Removes background and all other objects from a video, keeping only:
  • the single target object's silhouette (default), OR
  • the entire rectangular box around the target (new: KEEP_ENTIRE_BOX).

Dependencies:
    pip install ultralytics opencv-python numpy

Key features:
- Robust VideoWriter (handles ~120 FPS; tries multiple codecs/containers)
- ROI selection window shows full frame (scaled) and maps back to original
- Masks resized to exact frame size (fixes 1088 vs 1080 mismatch)
- Strict frame seek to your chosen start frame (works around VFR/MOV quirks)
- Warm-up acceptance to avoid early re-ID misses on moving cameras
- NEW: KEEP_ENTIRE_BOX with center/size smoothing (fixed size or tight to detection)
"""

import os
import cv2
import time
import shutil
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# ========= USER SETTINGS (EDIT HERE) =========================================
INPUT_VIDEO = r"E:\Evoq\Sprinting video\useful\IMG_2809.MOV"  # <--- your path

# Choose ONE selection method:
SELECTION_METHOD = "roi"    # "roi" or "class"
INIT_FRAME_INDEX = 78   # frame index to show for ROI and to start tracking

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


def main():
    # ========== Open video ==========
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {INPUT_VIDEO}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {INPUT_VIDEO} ({W}x{H} @ {FPS:.2f} FPS), frames={N_FRAMES}")

    # ========== Decide camera mode ==========
    cam_mode = CAMERA_MODE
    if cam_mode == "auto":
        cam_static = is_static_camera(cap)
        cam_mode = "static" if cam_static else "moving"
    print(f"[INFO] Camera mode: {cam_mode}")

    # ========== Load model ==========
    device_str = get_device_str()
    print(f"[INFO] Loading model: {MODEL_SIZE} on {device_str}")
    model = load_seg_model(MODEL_SIZE)
    predict_kwargs = dict(iou=NMS_IOU, conf=MIN_CONF, verbose=False,
                          imgsz=max(640, ((max(W,H)+31)//32)*32))

    # ========== Optional background subtractor ==========
    bg_sub = None
    if cam_mode == "static" and USE_BG_SUB_FOR_STATIC:
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
        print("[INFO] MOG2 background subtractor enabled (static camera).")

    # ========== Prepare first usable frame for selection ==========
    def get_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return ok, frame

    base_idx = max(0, min(INIT_FRAME_INDEX, N_FRAMES - 1))
    print(f"[INFO] INIT_FRAME_INDEX = {INIT_FRAME_INDEX} (selection frame)")
    ok, init_frame = get_frame(base_idx)
    if not ok:
        raise RuntimeError(f"Could not read frame {base_idx} for initialization.")

    # Detect on init frame
    init_results = model.predict(init_frame, **predict_kwargs)
    init_instances = masks_from_result(init_results[0], min_conf=MIN_CONF)
    if len(init_instances) == 0:
        raise RuntimeError("No instance found on the initialization frame. "
                           "Try a different frame index, lower MIN_CONF, or a larger model.")

    # Class maps
    class_id_to_name = getattr(model.model, "names", getattr(model, "names", {}))
    name_to_class_id = {v: k for k, v in class_id_to_name.items()}

    target_cls_id = None
    target_mask = None       # silhouette mask (for association/tracking)
    target_bbox = None
    target_id_text = None

    # For box mode
    init_roi_box = None      # (x1,y1,x2,y2)
    box_cx_f = None          # smoothed center x
    box_cy_f = None          # smoothed center y
    box_w_f  = None          # smoothed width
    box_h_f  = None          # smoothed height

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

        # Initialize box mode from first detection if requested
        if KEEP_ENTIRE_BOX:
            x1,y1,x2,y2 = target_bbox
            w0 = max(1, (x2-x1) + 2*BOX_PAD_PX)
            h0 = max(1, (y2-y1) + 2*BOX_PAD_PX)
            cx0 = (x1+x2)/2
            cy0 = (y1+y2)/2
            if BOX_MODE == "fixed_from_init":
                # Keep size fixed from the first detection
                box_w_f, box_h_f = float(w0), float(h0)
            else:
                # tight_to_detection: start with detection size; will update per-frame
                box_w_f, box_h_f = float(w0), float(h0)
            box_cx_f, box_cy_f = float(cx0), float(cy0)

    elif SELECTION_METHOD == "roi":
        # --- ROI window that shows full frame (scaled) ---
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

        # Pick the instance with max IoU with ROI (for class & tracking)
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

        # Initialize box mode using the ROI itself if requested
        if KEEP_ENTIRE_BOX:
            rx1, ry1, rx2, ry2 = init_roi_box
            w0 = rx2 - rx1
            h0 = ry2 - ry1
            cx0 = (rx1 + rx2) / 2.0
            cy0 = (ry1 + ry2) / 2.0
            if BOX_MODE == "fixed_from_init":
                box_w_f, box_h_f = float(w0), float(h0)  # fixed from ROI
            else:
                # tight_to_detection: start from detection size + pad
                x1,y1,x2,y2 = target_bbox
                box_w_f = float(max(1, (x2-x1) + 2*BOX_PAD_PX))
                box_h_f = float(max(1, (y2-y1) + 2*BOX_PAD_PX))
            box_cx_f, box_cy_f = float(cx0), float(cy0)
    else:
        raise RuntimeError("SELECTION_METHOD must be 'roi' or 'class'.")

    print(f"[INFO] Will start tracking from frame {base_idx}")

    # ========== Tracking state ==========
    prev_mask_f = None               # EMA buffer for OUTPUT mask (silhouette or box)
    last_assoc_mask = (target_mask>0).astype(np.uint8)  # for association only
    last_bbox = target_bbox
    misses = 0

    # Stats:
    t0 = time.time()
    frame_counter = 0

    if PREVIEW and PAUSE_ON_START:
        print("[INFO] Press any key in preview window to start...")
        cv2.imshow("preview", init_frame)
        cv2.waitKey(0)

    # ========== Writers ==========
    vw_masked, masked_path, FPS_OUT = make_video_writer(WRITE_MASKED, (W, H), FPS, is_color=True)
    vw_mask,   mask_path,   _      = make_video_writer(WRITE_MASK,   (W, H), FPS, is_color=False)

    # ========== Seek to the exact start frame ==========
    start_pos = seek_to_frame_strict(cap, base_idx, FPS)
    print(f"[INFO] Decode will start at frame {start_pos} (requested {base_idx})")

    print(f"[INFO] Tracking target: {target_id_text}")
    print(f"[INFO] Writing: {os.path.basename(masked_path)} and {os.path.basename(mask_path)}")

    # Main loop
    first_idx_printed = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not first_idx_printed:
            print(f"[INFO] First processed frame index: {idx}")
            first_idx_printed = True

        # SEGMENTATION
        results = model.predict(frame, **predict_kwargs)
        instances = masks_from_result(results[0], min_conf=MIN_CONF)

        # Optional BG refinement (static)
        fg_hint = None
        if bg_sub is not None:
            fg = bg_sub.apply(frame)
            fg = cv2.medianBlur(fg, 5)
            _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
            fg_hint = fg

        # Choose detection for our target (association using silhouette)
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

        # Accept or coast (with warm-up)
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
            # Update association state with the silhouette
            assoc_cur_mask = (picked["mask"]>0).astype(np.uint8)
            last_assoc_mask = assoc_cur_mask
            last_bbox = picked["bbox"]
            misses = 0

            # Update box center/size if enabled
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
                # else: fixed_from_init keeps size as-initialized

        else:
            # Coast
            misses += 1
            if misses == 1:
                print(f"[WARN] Miss at frame {idx} -> keeping previous mask/box")
            if misses > 0 and picked is not None:
                print(f"[INFO] Re-ID event at frame {idx}: score={best_score:.3f}")
            if misses > MAX_MISSES:
                print(f"[ERR ] Lost target for >{MAX_MISSES} frames. Stopping.")
                return  # early end

        # Build OUTPUT mask (silhouette or box)
        if KEEP_ENTIRE_BOX:
            # Ensure we have box params; if None (edge cases), initialize from last_bbox
            if box_cx_f is None or box_w_f is None:
                x1,y1,x2,y2 = last_bbox
                box_cx_f = (x1+x2)/2.0; box_cy_f = (y1+y2)/2.0
                if BOX_MODE == "tight_to_detection":
                    box_w_f = max(1, (x2-x1) + 2*BOX_PAD_PX)
                    box_h_f = max(1, (y2-y1) + 2*BOX_PAD_PX)
                else:
                    # fixed_from_init: if ROI existed, use its size; else use bbox+pad
                    if init_roi_box is not None:
                        rx1,ry1,rx2,ry2 = init_roi_box
                        box_w_f = float(rx2-rx1); box_h_f = float(ry2-ry1)
                    else:
                        box_w_f = max(1, (x2-x1) + 2*BOX_PAD_PX)
                        box_h_f = max(1, (y2-y1) + 2*BOX_PAD_PX)

            rect = rect_from_center_size(box_cx_f, box_cy_f, box_w_f, box_h_f, W, H)
            output_cur_mask = mask_from_rect(rect, H, W)
        else:
            # Default: use silhouette
            output_cur_mask = last_assoc_mask.copy()

        # Temporal smoothing (EMA) on OUTPUT mask
        prev_mask_f = ema_mask(prev_mask_f, output_cur_mask, alpha=MASK_SMOOTH_ALPHA)
        smoothed_u8 = (np.clip(prev_mask_f, 0, 1)*255).astype(np.uint8)

        # Post-process: morphology & feather
        post_mask_u8 = postprocess_mask((smoothed_u8 > 127).astype(np.uint8),
                                        FEATHER_PX, MORPH_OPEN, MORPH_CLOSE)

        # Safety resize
        if post_mask_u8.shape[:2] != frame.shape[:2]:
            post_mask_u8 = cv2.resize(post_mask_u8, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

        # Composite to black BG
        alpha = post_mask_u8.astype(np.float32) / 255.0
        alpha3 = np.dstack([alpha, alpha, alpha])
        out_masked = (frame.astype(np.float32) * alpha3).astype(np.uint8)

        # Write videos
        vw_masked.write(out_masked)
        vw_mask.write(post_mask_u8)

        # Optional PNG with alpha
        if SAVE_PNG_SEQUENCE:
            if not os.path.isdir(PNG_DIR):
                ensure_dir(PNG_DIR)
            b, g, r = cv2.split(frame)
            a = post_mask_u8
            rgba = cv2.merge([b, g, r, a])
            cv2.imwrite(os.path.join(PNG_DIR, f"frame_{idx:06d}.png"), rgba)

        # Preview
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
            if key == 27:   # ESC
                print("[INFO] ESC pressed. Stopping.")
                break
            if key == ord(' '):
                cv2.waitKey(0)

        # FPS stats
        frame_counter += 1
        if frame_counter % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            fps = frame_counter / max(1e-6, elapsed)
            print(f"[INFO] Processed {frame_counter} frames @ {fps:.2f} FPS")

    # Cleanup
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


if __name__ == "__main__":
    try:
        main()
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
