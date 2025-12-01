# pip install ultralytics opencv-python pandas
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ================= USER SETTINGS =================
VIDEO_PATH = r"E:\Evoq\Sprinting video\IMG_0751.MOV"
POSE_MODEL_PATH = "yolov8x-pose.pt"  # COCO-17 keypoints
START_FINISH_GAP_M = 10.0  # meters (real-world)
RUN_DIRECTION = "R2L"  # or "L2R" (for labeling only)
CONF_THR = 0.25
IOU_THR = 0.5
RESIZE_W, RESIZE_H = 1020, 500
SAVE_VIDEO = True
OUT_VIDEO = "bar_crossing_output.MOV"
OUT_CSV = "bar_crossing_speeds.csv"

EMA_ALPHA = 0.2  # hip smoothing
HYST_PIX = 6  # crossing hysteresis (px)


# =================================================

# ---------- Pose helpers ----------
def get_hip_xy_from_pose(model, frame):
    res = model.predict(frame, conf=CONF_THR, iou=IOU_THR, verbose=False)[0]
    if res.keypoints is None or res.boxes is None:
        return None
    kxy = res.keypoints.xy.cpu().numpy()  # (N,17,2)
    kcf = res.keypoints.conf.cpu().numpy()  # (N,17)
    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
    best = None
    best_conf = -1
    for i in range(len(cls)):
        if cls[i] != 0:
            continue
        c11, c12 = kcf[i][11], kcf[i][12]  # L/R hip conf
        if c11 > 0 and c12 > 0:
            lx, ly = kxy[i][11]
            rx, ry = kxy[i][12]
            hx, hy = (lx + rx) / 2.0, (ly + ry) / 2.0
            conf = min(c11, c12)
            if conf > best_conf:
                best_conf = conf
                best = (float(hx), float(hy))
    return best


# ---------- Draggable vertical bars ----------
class VerticalBarsUI:
    def __init__(self, img_w, img_h):
        self.w = img_w
        self.h = img_h
        # default positions (10% and 90% width)
        self.x_start = int(0.85 * img_w) if RUN_DIRECTION == "R2L" else int(0.15 * img_w)
        self.x_finish = int(0.15 * img_w) if RUN_DIRECTION == "R2L" else int(0.85 * img_w)
        self.active = None  # 'start' or 'finish' while dragging
        self.grab_margin = 12

    def draw(self, frame):
        # start line (red)
        cv2.line(frame, (self.x_start, 0), (self.x_start, self.h - 1), (0, 0, 255), 3)
        cv2.putText(frame, "START", (max(5, self.x_start - 60), 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # finish line (blue)
        cv2.line(frame, (self.x_finish, 0), (self.x_finish, self.h - 1), (255, 0, 0), 3)
        cv2.putText(frame, "FINISH", (min(self.w - 120, self.x_finish + 10), 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, "Drag lines to real 10m markers, ENTER to confirm",
                    (10, self.h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if abs(x - self.x_start) <= self.grab_margin:
                self.active = 'start'
            elif abs(x - self.x_finish) <= self.grab_margin:
                self.active = 'finish'
        elif event == cv2.EVENT_LBUTTONUP:
            self.active = None
        elif event == cv2.EVENT_MOUSEMOVE and self.active is not None:
            if self.active == 'start':
                self.x_start = max(0, min(self.w - 1, x))
            else:
                self.x_finish = max(0, min(self.w - 1, x))


def get_bar_positions(first_frame):
    ui = VerticalBarsUI(first_frame.shape[1], first_frame.shape[0])
    win = "Place Start/Finish"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, ui.on_mouse)

    while True:
        disp = first_frame.copy()
        ui.draw(disp)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # ENTER
            break
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise RuntimeError("Canceled.")
    cv2.destroyWindow(win)
    return ui.x_start, ui.x_finish


# ---------- Main ----------
def main():
    pose_model = YOLO(POSE_MODEL_PATH)
    pose_model.to("cuda")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        print("[WARN] Could not read FPS; defaulting to 30.")
        fps = 30.0

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame")
    first = cv2.resize(first, (RESIZE_W, RESIZE_H))

    # let user place bars
    x_start, x_finish = get_bar_positions(first)

    # reset stream to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (RESIZE_W, RESIZE_H))

    ema_hip = None
    have_start, have_finish = False, False
    start_f, finish_f = None, None
    f_idx = 0
    rows = []

    # For sign-change with hysteresis, track side values:
    last_side_start = None
    last_side_finish = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f_idx += 1
        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

        # draw bars
        cv2.line(frame, (x_start, 0), (x_start, RESIZE_H - 1), (0, 0, 255), 3)
        cv2.putText(frame, "START", (max(5, x_start - 60), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.line(frame, (x_finish, 0), (x_finish, RESIZE_H - 1), (255, 0, 0), 3)
        cv2.putText(frame, "FINISH", (min(RESIZE_W - 120, x_finish + 10), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0), 2)

        # pose -> hip
        hip = get_hip_xy_from_pose(pose_model, frame)
        if hip is not None:
            hx, hy = hip
            if ema_hip is None:
                ema_hip = (hx, hy)
            else:
                ema_hip = (EMA_ALPHA * hx + (1 - EMA_ALPHA) * ema_hip[0],
                           EMA_ALPHA * hy + (1 - EMA_ALPHA) * ema_hip[1])
            cv2.circle(frame, (int(ema_hip[0]), int(ema_hip[1])), 6, (0, 255, 255), -1)
        else:
            ema_hip = None

        # crossing detection (just in X since lines are vertical)
        if ema_hip is not None:
            ex, _ = ema_hip

            # START line sign (left = negative, right = positive)
            s_start = ex - x_start
            if last_side_start is None:
                last_side_start = s_start
            if not have_start:
                if (last_side_start < -HYST_PIX and s_start > HYST_PIX) or (
                        last_side_start > HYST_PIX and s_start < -HYST_PIX):
                    start_f = f_idx
                    have_start = True
            last_side_start = s_start

            # FINISH line, only after start was crossed
            s_finish = ex - x_finish
            if have_start and not have_finish:
                if last_side_finish is None:
                    last_side_finish = s_finish
                if (last_side_finish < -HYST_PIX and s_finish > HYST_PIX) or (
                        last_side_finish > HYST_PIX and s_finish < -HYST_PIX):
                    finish_f = f_idx
                    have_finish = True
                last_side_finish = s_finish

        # compute speed once per split
        if have_start and have_finish:
            frames_run = max(1, finish_f - start_f)
            elapsed = frames_run / fps
            v_ms = START_FINISH_GAP_M / elapsed
            v_kmh = v_ms * 3.6

            txt = f"{v_ms:.2f} m/s  ({v_kmh:.2f} km/h)  [{frames_run} frames @ {fps:.1f} fps]"
            cv2.rectangle(frame, (10, 10), (10 + 520, 60), (0, 0, 0), -1)
            cv2.putText(frame, txt, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            rows.append({
                "start_frame": start_f,
                "end_frame": finish_f,
                "frames": frames_run,
                "time_s": elapsed,
                "speed_m_s": v_ms,
                "speed_km_h": v_kmh,
                "direction": RUN_DIRECTION
            })

            # reset if you want to time multiple passes in same clip
            have_start = have_finish = False
            start_f = finish_f = None
            last_side_start = last_side_finish = None

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (RESIZE_W - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if SAVE_VIDEO and OUT_VIDEO:
            # lazy init writer after knowing final size
            if 'writer' not in locals() or writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (RESIZE_W, RESIZE_H))
            writer.write(frame)

        cv2.imshow("Runner Speed (Drag Bars)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if SAVE_VIDEO and 'writer' in locals() and writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"[INFO] Saved splits to {OUT_CSV}")
    else:
        print(
            "[INFO] No splits recorded. Make sure bars align with the true 10 m start/finish and the runner crosses both.")


if __name__ == "__main__":
    main()
