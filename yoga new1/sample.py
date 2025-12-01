import cv2
import numpy as np
import math
import tensorflow as tf
import tensorflow_hub as hub

# ========= USER SETTINGS =========
VIDEO_PATH = r"C:\Users\yenul\Downloads\pose identi - Made with Clipchamp.mp4" # change to your file
RECT_LEN_M = 10.0   # 10 m between start & finish cones (along run)
RECT_WIDTH_M = 2.0  # 2 m lane width (across cones)
SMOOTH_WINDOW = 5   # median window for light smoothing of Y(t) for OFFLINE check
SHOW_WARPED = False # optional top-down preview window
VIS_THR = 0.40      # keypoint confidence threshold for use
# =================================

# ---- Load MoveNet Thunder (singlepose) ----
# Thunder v4 expects 256x256 int32 input
print("[INFO] Loading MoveNet Thunder from TF Hub...")
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet_fn = movenet.signatures['serving_default']

# COCO-17 keypoint indices (y, x, score)
# 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear,
# 5 left shoulder, 6 right shoulder, 7 left elbow, 8 right elbow,
# 9 left wrist, 10 right wrist, 11 left hip, 12 right hip,
# 13 left knee, 14 right knee, 15 left ankle, 16 right ankle
COCO_LEFT_HIP, COCO_RIGHT_HIP = 11, 12
COCO_LEFT_KNEE, COCO_RIGHT_KNEE = 13, 14
COCO_LEFT_SH, COCO_RIGHT_SH = 5, 6

# ---------- UI helpers ----------
def click_points(win, frame, num=4, prompt="Click points"):
    pts = []
    clone = frame.copy()
    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < num:
            pts.append((x, y))
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _cb)
    while True:
        disp = clone.copy()
        cv2.putText(disp, f"{prompt} ({len(pts)}/{num})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        for i,p in enumerate(pts):
            cv2.circle(disp, p, 6, (0,0,255), -1)
            cv2.putText(disp, str(i+1), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            pts = []
            break
        if len(pts) == num:
            break
    cv2.destroyWindow(win)
    return np.array(pts, dtype=np.float32)

# ---------- Geometry / homography ----------
def build_homography(img_quad):
    # img_quad order:
    # 1) Start-L, 2) Start-R, 3) Finish-R, 4) Finish-L
    # Destination (meters): x across width, y along run
    dst = np.array([
        [0.0,            0.0],
        [RECT_WIDTH_M,   0.0],
        [RECT_WIDTH_M,   RECT_LEN_M],
        [0.0,            RECT_LEN_M],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(img_quad, dst, method=0)
    return H

def to_meters(H, pts_xy):
    pts = np.hstack([pts_xy, np.ones((pts_xy.shape[0],1))]).T  # 3xN
    mp_ = H @ pts
    mp_ /= (mp_[2,:] + 1e-12)
    return mp_[:2,:].T  # Nx2 (X_m, Y_m)

def to_image(Hinv, pts_m):
    pts = np.hstack([pts_m, np.ones((pts_m.shape[0],1))]).T
    ip = Hinv @ pts
    ip /= (ip[2,:] + 1e-12)
    return ip[:2,:].T  # Nx2 pixels

def smooth_series(xs, k=5):
    if k <= 1: return np.asarray(xs, dtype=float)
    k = int(k) + (int(k) % 2 == 0)
    half = k//2
    out = []
    xs = list(xs)
    for i in range(len(xs)):
        s = max(0, i-half)
        e = min(len(xs), i+half+1)
        out.append(np.median(xs[s:e]))
    return np.array(out, dtype=float)

def subframe_crossing(t0, t1, y0, y1, y_gate):
    dy = (y1 - y0)
    if dy == 0:
        return (t0 + t1) * 0.5, 0.5
    alpha = (y_gate - y0) / dy
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return t0 + alpha * (t1 - t0), alpha

# ---------- MoveNet inference & drawing ----------
def movenet_detect(frame_bgr):
    """Returns keypoints np.ndarray shape (17,3) with (y, x, score) normalized."""
    img = cv2.resize(frame_bgr, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = tf.convert_to_tensor(img, dtype=tf.int32)
    inp = tf.expand_dims(inp, axis=0)  # [1, 256, 256, 3]
    out = movenet_fn(inp)
    kps = out['output_0'].numpy()[0, 0, :, :]  # (17,3)
    return kps

# Nice-to-have skeleton for visualization
COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),               # shoulders-hips
    (11, 13), (13, 15), (12, 14), (14, 16)    # legs
]

def draw_skeleton(frame, kps, conf_thr=0.2):
    h, w = frame.shape[:2]
    # joints
    for i in range(17):
        y, x, c = kps[i]
        if c < conf_thr:
            continue
        cv2.circle(frame, (int(x*w), int(y*h)), 3, (0, 255, 0), -1)
    # bones
    for a, b in COCO_EDGES:
        ya, xa, ca = kps[a]
        yb, xb, cb = kps[b]
        if ca >= conf_thr and cb >= conf_thr:
            pa = (int(xa*w), int(ya*h))
            pb = (int(xb*w), int(yb*h))
            cv2.line(frame, pa, pb, (0, 180, 255), 2)

# ---------- Compute Y-along-run from keypoints ----------
def get_Y_along_run_m(H, kps, w, h, vis_thr=VIS_THR):
    """Return (Y_meters, draw_pt(x,y), tag) using robust hip/torso proxy (MoveNet keypoints)."""
    def ok(i): return kps[i, 2] > vis_thr
    def px(i): return (kps[i, 1] * w, kps[i, 0] * h)  # x = col, y = row

    # 1) mid-hip if both hips good
    if ok(COCO_LEFT_HIP) and ok(COCO_RIGHT_HIP):
        lx, ly = px(COCO_LEFT_HIP); rx, ry = px(COCO_RIGHT_HIP)
        cx, cy = (lx+rx)/2, (ly+ry)/2
        Y = to_meters(H, np.array([[cx, cy]], dtype=np.float32))[0][1]
        return float(Y), (int(cx), int(cy)), "mid_hip"

    # 2) torso center: avg shoulders and any hip center
    sh_ok = ok(COCO_LEFT_SH) and ok(COCO_RIGHT_SH)
    hip_any = ok(COCO_LEFT_HIP) or ok(COCO_RIGHT_HIP)
    if sh_ok and hip_any:
        sx1, sy1 = px(COCO_LEFT_SH); sx2, sy2 = px(COCO_RIGHT_SH)
        mxs, mys = (sx1+sx2)/2, (sy1+sy2)/2
        if ok(COCO_LEFT_HIP) and ok(COCO_RIGHT_HIP):
            hx1, hy1 = px(COCO_LEFT_HIP); hx2, hy2 = px(COCO_RIGHT_HIP)
            mxh, myh = (hx1+hx2)/2, (hy1+hy2)/2
        else:
            use = COCO_LEFT_HIP if kps[COCO_LEFT_HIP,2] >= kps[COCO_RIGHT_HIP,2] else COCO_RIGHT_HIP
            mxh, myh = px(use)
        cx, cy = (mxs+mxh)/2, (mys+myh)/2
        Y = to_meters(H, np.array([[cx, cy]], dtype=np.float32))[0][1]
        return float(Y), (int(cx), int(cy)), "torso_center"

    # 3) best single hip
    use = COCO_LEFT_HIP if kps[COCO_LEFT_HIP,2] >= kps[COCO_RIGHT_HIP,2] else COCO_RIGHT_HIP
    if ok(use):
        cx, cy = px(use)
        Y = to_meters(H, np.array([[cx, cy]], dtype=np.float32))[0][1]
        return float(Y), (int(cx), int(cy)), "single_hip"

    # 4) mid-knees as last resort
    if kps[COCO_LEFT_KNEE,2] > vis_thr and kps[COCO_RIGHT_KNEE,2] > vis_thr:
        kx1, ky1 = px(COCO_LEFT_KNEE); kx2, ky2 = px(COCO_RIGHT_KNEE)
        cx, cy = (kx1+kx2)/2, (ky1+ky2)/2
        Y = to_meters(H, np.array([[cx, cy]], dtype=np.float32))[0][1]
        return float(Y), (int(cx), int(cy)), "mid_knees"

    return None, None, "none"

# ---------- Main ----------
print(f"[INFO] OpenCV: {cv2.__version__}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 0 else 30.0
dt = 1.0 / fps

ok, first = cap.read()
if not ok:
    raise SystemExit("Could not read first frame.")

# Calibration clicks
quad = click_points("Mark 4 cone bases", first, 4,
                    prompt="Click 4 cone bases: Start-L, Start-R, Finish-R, Finish-L")
if quad is None or len(quad) != 4:
    raise SystemExit("Cone points not provided. Exiting.")

H = build_homography(quad)
Hinv = np.linalg.inv(H)

# Pre-compute gate lines in image space (two points each)
start_line_img = to_image(Hinv, np.array([[0.0, 0.0],
                                          [RECT_WIDTH_M, 0.0]], dtype=np.float32)).astype(int)
finish_line_img = to_image(Hinv, np.array([[0.0, RECT_LEN_M],
                                           [RECT_WIDTH_M, RECT_LEN_M]], dtype=np.float32)).astype(int)

# Optional top-down preview setup
if SHOW_WARPED:
    ppm = 80
    warped_size = (int(RECT_WIDTH_M*ppm), int(RECT_LEN_M*ppm))  # (w,h)
    S = np.array([[ppm, 0, 0],
                  [0,  ppm, 0],
                  [0,   0,  1]], dtype=np.float32)
    F = np.array([[1, 0, 0],
                  [0,-1, RECT_LEN_M],
                  [0, 0, 1]], dtype=np.float32)
    H_img2pix = S @ F @ H

# Rewind & process
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

times, y_along = [], []
frame_idx = 0

# Real-time state for visual clarity
state = "WAITING_START"  # WAITING_START -> TIMING -> DONE
t_start_vis = None
t_finish_vis = None
cross_pt_start = None
cross_pt_finish = None

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    t = frame_idx * dt  # use frame clock; swap to POS_MSEC if needed

    # ---- MoveNet Pose ----
    kps = movenet_detect(frame)  # (17,3)
    draw_skeleton(frame, kps, conf_thr=0.25)

    h, w = frame.shape[:2]
    Y_m, draw_pt, src_tag = get_Y_along_run_m(H, kps, w, h, vis_thr=VIS_THR)

    # Save series for offline calc
    if Y_m is not None and np.isfinite(Y_m):
        times.append(t)
        y_along.append(Y_m)

    # ---- Draw gate lines on the original view ----
    cv2.line(frame, tuple(start_line_img[0]), tuple(start_line_img[1]), (0,255,0), 3)   # start gate = green
    cv2.line(frame, tuple(finish_line_img[0]), tuple(finish_line_img[1]), (0,0,255), 3) # finish gate = red

    # Hip proxy point
    if draw_pt is not None:
        cv2.circle(frame, draw_pt, 6, (255,0,0), -1)  # blue = detected
    else:
        cv2.putText(frame, "Hip proxy not detected", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # ---- Real-time crossing visualization (using raw last two samples) ----
    if len(times) >= 2 and len(y_along) >= 2:
        t0, y0 = times[-2], y_along[-2]
        t1, y1 = times[-1], y_along[-1]

        # start crossing
        if state == "WAITING_START" and (y0 < 0.0 <= y1):
            t_cross, alpha = subframe_crossing(t0, t1, y0, y1, 0.0)
            cross_pt_start = draw_pt
            t_start_vis = t_cross
            state = "TIMING"

        # finish crossing
        if state == "TIMING" and (y0 < RECT_LEN_M <= y1):
            t_cross, alpha = subframe_crossing(t0, t1, y0, y1, RECT_LEN_M)
            cross_pt_finish = draw_pt
            t_finish_vis = t_cross
            state = "DONE"

    # Draw state & values
    cv2.putText(frame, f"State: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,220,255), 2)
    if Y_m is not None:
        cv2.putText(frame, f"Y={Y_m:.2f} m  src={src_tag}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    if t_start_vis is not None:
        cv2.putText(frame, f"Start @ {t_start_vis:.3f}s", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if cross_pt_start is not None:
            cv2.circle(frame, cross_pt_start, 10, (255,255,0), 2)

    if t_finish_vis is not None:
        cv2.putText(frame, f"Finish @ {t_finish_vis:.3f}s", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if cross_pt_finish is not None:
            cv2.circle(frame, cross_pt_finish, 10, (255,255,0), 2)

    # Optional top-down preview (visual only)
    if SHOW_WARPED:
        warped = cv2.warpPerspective(frame, H_img2pix, warped_size)
        cv2.line(warped, (0, int((RECT_LEN_M-0.0)*ppm)),
                         (warped.shape[1], int((RECT_LEN_M-0.0)*ppm)), (0,0,255), 2)
        cv2.line(warped, (0, int((RECT_LEN_M-RECT_LEN_M)*ppm)),
                         (warped.shape[1], int((RECT_LEN_M-RECT_LEN_M)*ppm)), (0,0,255), 2)
        cv2.imshow("Top-down (preview)", warped)

    cv2.imshow("Pose-based tracking (MoveNet Thunder)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---------- OFFLINE precise result (with light smoothing + sub-frame) ----------
times = np.array(times)
y_m = np.array(y_along)

def find_cross(times, ys, gate_value):
    ys_sm = smooth_series(ys, SMOOTH_WINDOW)
    for i in range(1, len(ys_sm)):
        if ys_sm[i-1] < gate_value <= ys_sm[i]:
            t_cross, _ = subframe_crossing(times[i-1], times[i], ys_sm[i-1], ys_sm[i], gate_value)
            return t_cross
    return None

if len(times) < 3:
    print("[ERROR] Not enough valid pose detections to compute speed.")
else:
    t_start  = find_cross(times, y_m, 0.0)
    t_finish = find_cross(times, y_m, RECT_LEN_M)

    if t_start is None or t_finish is None or t_finish <= t_start:
        print("[ERROR] Could not determine valid start/finish crossings. "
              "Re-check cone clicks and that the athlete traverses the full 10 m zone.")
    else:
        fps_nominal = fps
        delta_t = t_finish - t_start
        v_ms = RECT_LEN_M / delta_t
        v_kmh = v_ms * 3.6

        # Uncertainty estimate (conservative)
        timing_sigma = (0.25 / fps_nominal) * math.sqrt(2)
        rel_sigma = timing_sigma / delta_t
        v_sigma = v_ms * rel_sigma

        print(f"Start crossing:  {t_start:.4f} s")
        print(f"Finish crossing: {t_finish:.4f} s")
        print(f"Δt: {delta_t:.4f} s")
        print(f"Speed: {v_ms:.3f} m/s  ({v_kmh:.2f} km/h)")
        print(f"Estimated uncertainty: ±{v_sigma:.3f} m/s  (±{v_sigma*3.6:.2f} km/h) "
              f"~{rel_sigma*100:.2f}%")
