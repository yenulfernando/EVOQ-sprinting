# pip install ultralytics opencv-python pandas numpy

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import *   # uses your centroid-based Tracker

# ================= USER SETTINGS =================
VIDEO_PATH = r"E:\Evoq\Sprinting video\IMG_0751.MOV"  # <-- your sprint video
MODEL_PATH = "yolov8x.pt"          # COCO detect model (person class id = 0)
RUN_DIRECTION = "R2L"              # "R2L" or "L2R"
START_FINISH_GAP_M = 10.0          # meters between start & finish (real-world)
CONF_THR = 0.25
IOU_THR  = 0.5
RESIZE_W, RESIZE_H = 1020, 500
SAVE_VIDEO = True
OUT_VIDEO  = "sprinter_output_calibrated.MOV"
OUT_CSV    = "sprinter_splits_calibrated.csv"
# ==================================================

def center_of_box(x1, y1, x2, y2):
    return int((x1 + x2) // 2), int((y1 + y2) // 2)

# ========== STEP 1: Calibration function ==========
def calibrate_camera(video_path, resize_w, resize_h):
    print("\n📏 Calibration: Click 4 ground points in order (bottom-left → bottom-right → top-right → top-left)")
    calibration_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration_points.append([x, y])
            print(f"Point {len(calibration_points)}: ({x}, {y})")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read video frame for calibration.")
    frame = cv2.resize(frame, (resize_w, resize_h))

    cv2.imshow("Calibration - Click 4 points on ground (then press any key)", frame)
    cv2.setMouseCallback("Calibration - Click 4 points on ground (then press any key)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(calibration_points) != 4:
        raise ValueError("You must click exactly 4 points on the ground.")

    # Corresponding real-world coordinates (meters)
    # Adjust this rectangle according to actual track layout
    world_points = np.array([
        [0, 0],
        [10, 0],   # e.g., 10 m width between left-right clicks
        [10, 2],
        [0, 2]
    ], dtype=np.float32)

    image_points = np.array(calibration_points, dtype=np.float32)
    H, _ = cv2.findHomography(image_points, world_points)
    print("✅ Homography matrix computed.\n")
    return H

# ==================================================

def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    model.to("cuda")

    # --- Calibrate camera once before detection ---
    H = calibrate_camera(VIDEO_PATH, RESIZE_W, RESIZE_H)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    # Video writer
    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (RESIZE_W, RESIZE_H))

    tracker = Tracker()

    start_frame_by_id = {}
    counted_forward = set()
    last_real_x = {}

    # Visuals
    red_color, blue_color, text_color = (0,0,255), (255,0,0), (0,0,0)
    box_color, center_color, hud_bg = (60,200,60), (0,0,255), (0,255,255)

    frame_idx = 0
    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

        # --- Detect people ---
        result = model.predict(frame, conf=CONF_THR, iou=IOU_THR, verbose=False)[0]
        det_list = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                if cls[i] == 0:  # person
                    det_list.append(list(map(int, xyxy[i][:4])))

        # --- Track IDs ---
        tracks = tracker.update(det_list)

        # HUD background
        cv2.rectangle(frame, (0, 0), (300, 100), hud_bg, -1)

        for x1, y1, x2, y2, tid in tracks:
            cx, cy = center_of_box(x1, y1, x2, y2)
            pt = np.array([[cx, cy]], dtype=np.float32).reshape(-1,1,2)
            real_pt = cv2.perspectiveTransform(pt, H)[0][0]
            real_x, real_y = real_pt[0], real_pt[1]

            # Draw bounding box & id
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, max(20, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.circle(frame, (cx, cy), 4, center_color, -1)

            if tid in last_real_x:
                moving_left = (real_x - last_real_x[tid]) < 0
            else:
                moving_left = False
            last_real_x[tid] = real_x

            if RUN_DIRECTION == "R2L":
                forward_moving = moving_left
                start_pos, finish_pos = 10.0, 0.0
            else:
                forward_moving = not moving_left
                start_pos, finish_pos = 0.0, 10.0

            # Detect crossing start line
            if forward_moving and tid not in start_frame_by_id and abs(real_x - start_pos) < 0.2:
                start_frame_by_id[tid] = frame_idx

            # Detect crossing finish line
            if tid in start_frame_by_id and tid not in counted_forward and abs(real_x - finish_pos) < 0.2:
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
                        "start_frame": start_frame_by_id[tid],
                        "end_frame": frame_idx,
                        "time_s": elapsed,
                        "speed_m_s": v_ms,
                        "speed_km_h": v_kmh
                    })

        cv2.putText(frame, f"Detections: {len(tracks)}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        cv2.putText(frame, f"Splits: {len(counted_forward)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        if SAVE_VIDEO:
            writer.write(frame)

        cv2.imshow("Sprinter Speed Detection (Calibrated)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if SAVE_VIDEO and writer:
        writer.release()
    cv2.destroyAllWindows()

    # Save results
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"\n✅ Saved real-world calibrated splits to {OUT_CSV}")
    else:
        print("\n[INFO] No splits recorded. Check calibration and athlete path.")

if __name__ == "__main__":
    main()
