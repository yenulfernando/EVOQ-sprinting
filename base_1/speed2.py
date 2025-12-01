# pip install ultralytics opencv-python pandas

import os
import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *   # uses your centroid-based Tracker

# ================= USER SETTINGS =================
VIDEO_PATH = r"E:\Evoq\Sprinting video\useful\IMG_2809.MOV" # <-- your sprint video
MODEL_PATH = "yolov8x.pt"          # COCO detect model (person class id = 0)

# Choose direction of the runner: "R2L" (Right to Left) or "L2R" (Left to Right)
RUN_DIRECTION = "R2L"

ORIENTATION = "x"                  # keep as "x" for vertical start/finish lines
START_FINISH_GAP_M = 10.0          # meters between start & finish (real-world)

CONF_THR = 0.25
IOU_THR  = 0.5

RESIZE_W, RESIZE_H = 1020, 500     # output display resolution (adjust if you want)
SAVE_VIDEO = True
OUT_VIDEO  = "sprinter_output.MOV"
SAVE_FRAMES = False                # set True if you want frame-by-frame JPGs
OUT_CSV    = "sprinter_splits.csv" # split results table
# ==================================================

def center_of_box(x1, y1, x2, y2):
    return int((x1 + x2) // 2), int((y1 + y2) // 2)

def main():
    # Load model
    model = YOLO(MODEL_PATH)
    model.to("cuda")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        print("[WARN] Could not read FPS from file; defaulting to 30.")
        fps = 30.0

    # Prepare writer
    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (RESIZE_W, RESIZE_H))

    # Optional frames folder
    if SAVE_FRAMES and not os.path.exists("detected_frames"):
        os.makedirs("detected_frames")

    tracker = Tracker()

    # Book-keeping (frame-based timing)
    start_frame_by_id = {}
    end_frame_by_id   = {}
    counted_forward   = set()
    counted_reverse   = set()
    last_x = {}

    # Visuals
    text_color  = (0, 0, 0)
    red_color   = (0, 0, 255)   # start
    blue_color  = (255, 0, 0)   # finish
    box_color   = (60, 200, 60)
    center_color= (0, 0, 255)
    hud_bg      = (0, 255, 255)
    offset = 6

    # ---------------- Set vertical lines depending on direction ----------------
    if RUN_DIRECTION == "R2L":
        start_line  = RESIZE_W - 100   # right side
        finish_line = 100              # left side
        start_label, finish_label = "Start (Right)", "Finish (Left)"
    else:  # L2R
        start_line  = 100              # left side
        finish_line = RESIZE_W - 100   # right side
        start_label, finish_label = "Start (Left)", "Finish (Right)"
    # ---------------------------------------------------------------------------

    frame_idx = 0
    saved_idx = 0
    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

        # Detect only 'person'
        result = model.predict(frame, conf=CONF_THR, iou=IOU_THR, verbose=False)[0]
        det_list = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            cls  = result.boxes.cls.detach().cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                if cls[i] == 0:  # COCO 'person'
                    x1, y1, x2, y2 = map(int, xyxy[i][:4])
                    det_list.append([x1, y1, x2, y2])

        # Track
        tracks = tracker.update(det_list)

        # HUD panel
        cv2.rectangle(frame, (0, 0), (300, 110), hud_bg, -1)

        # Draw vertical lines + labels
        cv2.line(frame, (start_line, 0), (start_line, RESIZE_H-1), red_color, 2)
        cv2.putText(frame, start_label, (max(5, start_line-120), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        cv2.line(frame, (finish_line, 0), (finish_line, RESIZE_H-1), blue_color, 2)
        cv2.putText(frame, finish_label, (finish_line+6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # Per tracked person
        for x1, y1, x2, y2, tid in tracks:
            cx, cy = center_of_box(x1, y1, x2, y2)

            # Visualize box, id, center
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, max(20, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.circle(frame, (cx, cy), 4, center_color, -1)

            # Direction check
            moving_left = False
            if tid in last_x:
                moving_left = (cx - last_x[tid]) < 0
            last_x[tid] = cx

            if RUN_DIRECTION == "R2L":
                forward_moving = moving_left
            else:  # L2R
                forward_moving = not moving_left

            # Crossing checks
            start_hit  = (start_line  < (cx + offset)) and (start_line  > (cx - offset))
            finish_hit = (finish_line < (cx + offset)) and (finish_line > (cx - offset))

            # Forward run timing
            if start_hit and forward_moving and tid not in start_frame_by_id:
                start_frame_by_id[tid] = frame_idx

            if (tid in start_frame_by_id) and finish_hit and (tid not in counted_forward):
                frames = frame_idx - start_frame_by_id[tid]
                if frames > 0:
                    elapsed = frames / fps
                    v_ms  = START_FINISH_GAP_M / elapsed
                    v_kmh = v_ms * 3.6
                    counted_forward.add(tid)

                    cv2.putText(frame, f"{v_kmh:.2f} km/h", (x2, y2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                    rows.append({
                        "id": tid,
                        "direction": "R2L" if RUN_DIRECTION == "R2L" else "L2R",
                        "start_frame": start_frame_by_id[tid],
                        "end_frame": frame_idx,
                        "frames": frames,
                        "time_s": elapsed,
                        "speed_m_s": v_ms,
                        "speed_km_h": v_kmh
                    })

            # Optional reverse run (back the other way)
            if finish_hit and not forward_moving and tid not in end_frame_by_id:
                end_frame_by_id[tid] = frame_idx

            if (tid in end_frame_by_id) and start_hit and (tid not in counted_reverse):
                frames = frame_idx - end_frame_by_id[tid]
                if frames > 0:
                    elapsed = frames / fps
                    v_ms  = START_FINISH_GAP_M / elapsed
                    v_kmh = v_ms * 3.6
                    counted_reverse.add(tid)

                    cv2.putText(frame, f"{v_kmh:.2f} km/h", (x2, y2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                    rows.append({
                        "id": tid,
                        "direction": "L2R" if RUN_DIRECTION == "R2L" else "R2L",
                        "start_frame": end_frame_by_id[tid],
                        "end_frame": frame_idx,
                        "frames": frames,
                        "time_s": elapsed,
                        "speed_m_s": v_ms,
                        "speed_km_h": v_kmh
                    })

        # Small HUD text
        cv2.putText(frame, f"Detections: {len(tracks)}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        cv2.putText(frame, f"Splits Forward: {len(counted_forward)}  Reverse: {len(counted_reverse)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # Save/show
        if SAVE_VIDEO:
            writer.write(frame)
        if SAVE_FRAMES:
            saved_idx += 1
            cv2.imwrite(f"detected_frames/frame_{saved_idx:06d}.jpg", frame)

        cv2.imshow("Sprinter Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if SAVE_VIDEO and writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Save CSV
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        if os.path.exists(OUT_CSV):
            old = pd.read_csv(OUT_CSV)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"[INFO] Saved splits to {OUT_CSV}")
    else:
        print("[INFO] No splits recorded. Check your line positions and the athlete’s path.")

if __name__ == "__main__":
    main()
