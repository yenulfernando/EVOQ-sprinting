# pip install ultralytics opencv-python pandas
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *   # uses your centroid-based Tracker

# ================= USER SETTINGS =================
VIDEO_PATH = r"E:\Evoq\Sprinting video\IMG_0751.MOV"
MODEL_PATH = "yolov8x.pt"  # COCO detect model (person class id = 0)

RUN_DIRECTION = "R2L"          # or "L2R"
START_FINISH_GAP_M = 10.0      # real-world distance in meters
CONF_THR = 0.25
IOU_THR  = 0.5
RESIZE_W, RESIZE_H = 1020, 500
SAVE_VIDEO = True
OUT_VIDEO  = "sprinter_output.MOV"
OUT_CSV    = "sprinter_splits.csv"
# ==================================================

# ---------- Draggable vertical bars ----------
class VerticalBarsUI:
    def __init__(self, img_w, img_h, run_direction):
        self.w, self.h = img_w, img_h
        self.run_direction = run_direction
        self.x_start = int(0.85 * img_w) if run_direction == "R2L" else int(0.15 * img_w)
        self.x_finish = int(0.15 * img_w) if run_direction == "R2L" else int(0.85 * img_w)
        self.active = None
        self.grab_margin = 12

    def draw(self, frame):
        cv2.line(frame, (self.x_start, 0), (self.x_start, self.h-1), (0,0,255), 3)
        cv2.putText(frame, "START", (max(5, self.x_start-60), 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.line(frame, (self.x_finish, 0), (self.x_finish, self.h-1), (255,0,0), 3)
        cv2.putText(frame, "FINISH", (min(self.w-120, self.x_finish+10), 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, "Drag lines to match 10 m markers • Press ENTER to confirm",
                    (10, self.h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if abs(x - self.x_start) <= self.grab_margin:
                self.active = "start"
            elif abs(x - self.x_finish) <= self.grab_margin:
                self.active = "finish"
        elif event == cv2.EVENT_LBUTTONUP:
            self.active = None
        elif event == cv2.EVENT_MOUSEMOVE and self.active:
            if self.active == "start":
                self.x_start = max(0, min(self.w-1, x))
            else:
                self.x_finish = max(0, min(self.w-1, x))

def get_bar_positions(first_frame, run_direction):
    ui = VerticalBarsUI(first_frame.shape[1], first_frame.shape[0], run_direction)
    win = "Set Start/Finish Lines"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, ui.on_mouse)

    while True:
        disp = first_frame.copy()
        ui.draw(disp)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:   # ENTER
            break
        if key == 27:   # ESC
            cv2.destroyWindow(win)
            raise RuntimeError("Canceled by user")
    cv2.destroyWindow(win)
    return ui.x_start, ui.x_finish
# --------------------------------------------------

def center_of_box(x1, y1, x2, y2):
    return int((x1 + x2) // 2), int((y1 + y2) // 2)

def main():
    model = YOLO(MODEL_PATH).to("cuda")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    # === interactive placement on first frame ===
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame")
    first = cv2.resize(first, (RESIZE_W, RESIZE_H))
    start_line, finish_line = get_bar_positions(first, RUN_DIRECTION)

    # reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (RESIZE_W, RESIZE_H))

    tracker = Tracker()
    start_frame_by_id, end_frame_by_id = {}, {}
    counted_forward, counted_reverse, last_x = set(), set(), {}

    frame_idx = 0
    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

        # detect
        res = model.predict(frame, conf=CONF_THR, iou=IOU_THR, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            cls  = res.boxes.cls.detach().cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                if cls[i] == 0:
                    x1,y1,x2,y2 = map(int, xyxy[i][:4])
                    dets.append([x1,y1,x2,y2])

        tracks = tracker.update(dets)

        # draw bars
        cv2.line(frame, (start_line,0), (start_line,RESIZE_H-1), (0,0,255), 2)
        cv2.putText(frame, "START", (max(5,start_line-60),25), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.line(frame, (finish_line,0), (finish_line,RESIZE_H-1), (255,0,0), 2)
        cv2.putText(frame, "FINISH", (min(RESIZE_W-120,finish_line+10),25), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

        # per ID
        for x1,y1,x2,y2,tid in tracks:
            cx, cy = center_of_box(x1,y1,x2,y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(60,200,60),2)
            cv2.putText(frame,f"ID {tid}",(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.circle(frame,(cx,cy),4,(0,255,255),-1)

            moving_left = False
            if tid in last_x:
                moving_left = (cx - last_x[tid]) < 0
            last_x[tid] = cx
            forward = moving_left if RUN_DIRECTION=="R2L" else not moving_left

            start_hit  = (start_line  < cx+6) and (start_line  > cx-6)
            finish_hit = (finish_line < cx+6) and (finish_line > cx-6)

            # forward timing
            if start_hit and forward and tid not in start_frame_by_id:
                start_frame_by_id[tid] = frame_idx

            if tid in start_frame_by_id and finish_hit and tid not in counted_forward:
                frames = frame_idx - start_frame_by_id[tid]
                if frames > 0:
                    elapsed = frames / fps
                    v_ms = START_FINISH_GAP_M / elapsed
                    v_kmh = v_ms * 3.6
                    counted_forward.add(tid)
                    cv2.putText(frame, f"{v_kmh:.2f} km/h", (x2, y2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    rows.append({
                        "id": tid, "direction": RUN_DIRECTION,
                        "start_frame": start_frame_by_id[tid],
                        "end_frame": frame_idx,
                        "frames": frames, "time_s": elapsed,
                        "speed_m_s": v_ms, "speed_km_h": v_kmh
                    })

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(frame, f"Splits: {len(rows)}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

        if SAVE_VIDEO:
            writer.write(frame)
        cv2.imshow("Sprinter Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if SAVE_VIDEO and writer:
        writer.release()
    cv2.destroyAllWindows()

    if rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"[INFO] Results saved to {OUT_CSV}")
    else:
        print("[INFO] No splits recorded — ensure runner crosses both lines.")
if __name__ == "__main__":
    main()
