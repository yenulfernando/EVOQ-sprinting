"""
Keep-One-Object Video Compositor (Updated + Strict Seek + Warm-up)
------------------------------------------------------------------
Removes background and all other objects from a video, keeping only a single target
object visible across frames.

Dependencies:
    pip install ultralytics opencv-python numpy

Highlights:
- Robust VideoWriter (handles ~120 FPS timebase; tries multiple codecs/containers)
- ROI selection window shows full frame (scaled to screen) and maps back to original
- Segmentation masks are resized to exact frame size (fixes 1088 vs 1080 mismatch)
- Strict frame seek to your chosen start frame (works around VFR/MOV quirks)
- Warm-up acceptance window to avoid early re-ID misses on moving cameras
- Safety size checks before compositing

Outputs:
- output_masked.mp4 / .avi : only the chosen object, background black
- output_mask.mp4   / .avi : grayscale mask (white = keep)
- frames_rgba/ (optional)  : PNG sequence with alpha (mask)
"""

import os
import cv2
import time
import shutil
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# ========= USER SETTINGS (EDIT HERE) =========================================
INPUT_VIDEO = r"E:\Evoq\Sprinting video\useful\Untitled video slo mo - Made with Clipchamp.mp4"  # <--- your path

# Choose ONE selection method:
SELECTION_METHOD = "roi"   # "roi" or "class"
INIT_FRAME_INDEX = 63     # frame index to show for ROI (0-based) AND to start tracking from

# If SELECTION_METHOD == "class":
CLASS_LABEL = "person"     # e.g., "person", "dog", "car", ...

# Camera mode:
CAMERA_MODE = "auto"       # "auto" | "static" | "moving"
USE_BG_SUB_FOR_STATIC = True  # If static camera, use MOG2 to refine/assist

# Model selection:
MODEL_SIZE = "yolov8x-seg"   # "yolov8n-seg" (faster), "yolov8s-seg", "yolov8m-seg", ...
MIN_CONF = 0.25              # discard low-confidence detections
NMS_IOU = 0.5                # NMS IoU for detections

# Tracking / Re-ID:
IOU_SWITCH_THRESH = 0.25     # avoid ID switch (0.15–0.30 typical)
MAX_MISSES = 30              # how many frames we allow to "coast" when lost
REID_MIN_OVERLAP = 0.05      # needed mask overlap to "re-identify"

# Warm-up acceptance for first frames after start (helps moving cameras)
WARMUP_FRAMES = 10           # frames with slightly looser acceptance
WARMUP_IOU_THR = 0.12        # relaxed IoU / overlap threshold during warm-up
WARMUP_BEST_SCORE = 0.10     # relaxed best_score acceptance during warm-up
STEADY_BEST_SCORE = 0.15     # best_score acceptance after warm-up

# Mask smoothing & post-process:
MASK_SMOOTH_ALPHA = 0.35     # EMA smoothing [0..1], higher = more weight on new
MORPH_OPEN = 3               # kernel size (0 disables)
MORPH_CLOSE = 5              # kernel size (0 disables)
FEATHER_PX = 6               # Gaussian feather in pixels

# Export (paths are used as bases; codec may switch extension for compatibility)
WRITE_MASKED = "output_masked.mp4"
WRITE_MASK = "output_mask.mp4"
SAVE_PNG_SEQUENCE = False        # if True, save frames_rgba/ with alpha
PNG_DIR = "frames_rgba"

# Preview window:
PREVIEW = True                   # show live window
PREVIEW_INPAINT_COMPARE = False  # show crude inpaint comparison side-by-side
PAUSE_ON_START = False           # pause on first rendered frame

# ROI display size (use your monitor size here; the frame is scaled to fit)
ROI_WINDOW_MAX = (1280, 720)     # (width, height)
PRINT_EVERY = 30                 # print FPS stats every N frames
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
    """
    Create a robust VideoWriter with sensible fallbacks for high-FPS videos.
    Returns (writer, out_path, fps_out).
    """
    W, H = size
    # Many backends dislike fractional FPS like 119.95; round to int and clamp
    fps_out = int(round(min(120, max(1, fps_in))))

    trials = [
        ("mp4v", ".mp4"),  # MPEG-4 Part 2
        ("avc1", ".mp4"),  # H.264 (if OpenCV build supports)
        ("H264", ".mp4"),  # alternate H.264 code
        ("XVID", ".avi"),  # AVI fallback
        ("MJPG", ".avi"),  # Large but very compatible
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
    raise RuntimeError("Failed to initialize VideoWriter with all tested codecs/containers.")


def masks_from_result(result, min_conf=0.25) -> List[Dict[str, Any]]:
    """
    Extract instance segmentation masks from a single Ultralytics result.
    Always returns masks resized to the original frame size (H, W).
    Each item: {"mask": uint8 {0,1} at (H,W), "bbox": (x1,y1,x2,y2), "cls": int, "conf": float}
    """
    out = []
    if result.masks is None or result.boxes is None:
        return out

    H, W = result.orig_shape  # (height, width) of the input frame
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
        # Ensure mask is HxW
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
    """
    Try to seek to an exact frame index. If OpenCV can't land precisely,
    read frames until we reach the target (or give up after max_step_read).
    Returns the actual starting frame index it reached.
    """
    target_idx = max(0, target_idx)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

    # Some backends don't update until a read
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if pos != target_idx:
        ok, _ = cap.read()
        if ok:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            return int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # If we're still behind, fast-forward
    steps = 0
    while pos < target_idx and steps < max_step_read:
        ok, _ = cap.read()
        if not ok:
            break
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        steps += 1
    return pos


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

    # ========== Optional background subtractor for static camera ==========
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

    # Run detection on init frame
    init_results = model.predict(init_frame, **predict_kwargs)
    init_instances = masks_from_result(init_results[0], min_conf=MIN_CONF)

    if len(init_instances) == 0:
        raise RuntimeError(
            "No instance found on the initialization frame. "
            "Try a different frame index, lower MIN_CONF, or a larger model."
        )

    # Class maps
    class_id_to_name = getattr(model.model, "names", getattr(model, "names", {}))
    name_to_class_id = {v: k for k, v in class_id_to_name.items()}

    target_cls_id = None
    target_mask = None
    target_bbox = None
    target_id_text = None

    if SELECTION_METHOD == "class":
        if CLASS_LABEL not in name_to_class_id:
            raise RuntimeError(
                f"CLASS_LABEL '{CLASS_LABEL}' not in model classes. "
                f"Available: {sorted(set(class_id_to_name.values()))[:20]} ..."
            )
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

    elif SELECTION_METHOD == "roi":
        # --- ROI window that shows full frame (scaled) ---
        src_h, src_w = init_frame.shape[:2]
        max_w, max_h = ROI_WINDOW_MAX
        scale_w = max_w / src_w
        scale_h = max_h / src_h
        scale = min(1.0, min(scale_w, scale_h))  # never upscale; just fit-to-screen
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

        # Map ROI back to original coordinates
        rx = int(rx / scale); ry = int(ry / scale)
        rw = int(rw / scale); rh = int(rh / scale)
        roi_box = (rx, ry, rx+rw, ry+rh)

        # Pick the instance with max IoU with ROI
        best = None
        best_iou = -1.0
        for ins in init_instances:
            iou = iou_box(ins["bbox"], roi_box)
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

    else:
        raise RuntimeError("SELECTION_METHOD must be 'roi' or 'class'.")

    print(f"[INFO] Will start tracking from frame {base_idx}")

    # ========== Tracking state ==========
    prev_mask_f = None         # float EMA (0..1)
    last_good_mask = (target_mask>0).astype(np.uint8)
    last_bbox = target_bbox
    misses = 0

    # For FPS stats:
    t0 = time.time()
    frame_counter = 0

    if PREVIEW and PAUSE_ON_START:
        print("[INFO] Press any key in preview window to start...")
        cv2.imshow("preview", init_frame)
        cv2.waitKey(0)

    # ========== Output writers (robust) ==========
    # Create writers AFTER we know W,H,FPS and before main loop
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

        # Decoder-reported frame index (no -1 confusion)
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

        # Choose the detection for our target:
        picked = None
        best_score = -1.0

        # 1) Prefer same class id (if known)
        for ins in instances:
            if target_cls_id is not None and ins["cls"] != target_cls_id:
                continue
            score = 0.0
            if last_bbox is not None:
                score += 0.7 * iou_box(ins["bbox"], last_bbox)
            if last_good_mask is not None:
                inter = overlap_ratio(last_good_mask, ins["mask"])
                score += 0.3 * inter
            if fg_hint is not None:
                fg_overlap = overlap_ratio((fg_hint > 0).astype(np.uint8), ins["mask"])
                score += 0.15 * fg_overlap
            if score > best_score:
                best_score = score
                picked = ins

        # 2) Re-ID fallback
        if picked is None and instances:
            for ins in instances:
                inter = overlap_ratio(last_good_mask, ins["mask"]) if last_good_mask is not None else 0.0
                if inter > REID_MIN_OVERLAP and inter > best_score:
                    best_score = inter
                    picked = ins

        # Accept or coast, with warm-up logic
        accepted = False
        relax = (idx - start_pos) < WARMUP_FRAMES
        thr = WARMUP_IOU_THR if relax else IOU_SWITCH_THRESH
        best_thr = WARMUP_BEST_SCORE if relax else STEADY_BEST_SCORE

        if picked is not None:
            if last_bbox is None:
                accepted = True
            else:
                iou = iou_box(picked["bbox"], last_bbox)
                inter = overlap_ratio(last_good_mask, picked["mask"]) if last_good_mask is not None else 0.0
                if (iou >= thr) or (inter >= thr) or (best_score >= best_thr):
                    accepted = True

        if accepted:
            cur_mask = (picked["mask"]>0).astype(np.uint8)
            last_good_mask = cur_mask
            last_bbox = picked["bbox"]
            misses = 0
        else:
            misses += 1
            cur_mask = last_good_mask if last_good_mask is not None else np.zeros((H,W), np.uint8)
            if misses == 1:
                print(f"[WARN] Miss at frame {idx} -> keeping previous mask")
            if misses > 0 and picked is not None:
                print(f"[INFO] Re-ID event at frame {idx}: score={best_score:.3f}")
            if misses > MAX_MISSES:
                print(f"[ERR ] Lost target for >{MAX_MISSES} frames. Stopping.")
                break

        # Temporal smoothing (EMA on binary mask)
        prev_mask_f = ema_mask(prev_mask_f, cur_mask, alpha=MASK_SMOOTH_ALPHA)
        smoothed_u8 = (np.clip(prev_mask_f, 0, 1)*255).astype(np.uint8)

        # Post-process: morphology & feather
        post_mask_u8 = postprocess_mask((smoothed_u8 > 127).astype(np.uint8),
                                        FEATHER_PX, MORPH_OPEN, MORPH_CLOSE)

        # Safety: ensure mask matches frame size exactly
        if post_mask_u8.shape[:2] != frame.shape[:2]:
            post_mask_u8 = cv2.resize(post_mask_u8, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

        # Composite: foreground only; background black
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
            if key == 27:  # ESC
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
        print("- If tracking drifts:")
        print("    * Lower IOU_SWITCH_THRESH to 0.15–0.20")
        print("    * Increase MASK_SMOOTH_ALPHA to stabilize mask")
        print("    * Increase MORPH_CLOSE or FEATHER_PX to soften edge flicker")
        print("- If camera is static, keep USE_BG_SUB_FOR_STATIC=True for speed/robustness.")
