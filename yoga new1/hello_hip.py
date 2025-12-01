import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import argparse

def track_hip_x(video_path, min_det=0.5, min_trk=0.5, model_complexity=1):
    """Return (hip_x_px_per_frame, frame_width_px). NaN where hip not found."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_trk
    )

    xs = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)

            hip_x_px = np.nan
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                try:
                    L = lm[mp_pose.PoseLandmark.LEFT_HIP]
                    R = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                    if L.visibility > 0.4 and R.visibility > 0.4:
                        hip_x_px = ((L.x + R.x) * 0.5) * width
                except Exception:
                    hip_x_px = np.nan

            xs.append(float(hip_x_px))
    finally:
        cap.release()
        pose.close()

    return np.array(xs, dtype=float), width

def _resolve_csv_path(video_path, requested_csv_path):
    """Pick a safe CSV output path; fallback to CWD if needed. Returns absolute path."""
    if requested_csv_path:
        out_path = requested_csv_path
    else:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(os.path.dirname(video_path), f"{base}_hip_x.csv")

    out_dir = os.path.dirname(out_path) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.abspath(out_path)
    except Exception:
        # Fallback to current working directory
        fallback = os.path.abspath(os.path.join(os.getcwd(), os.path.basename(out_path)))
        print(f"[WARN] Could not create or write to '{out_dir}'. Falling back to: {fallback}")
        return fallback

def main():
    parser = argparse.ArgumentParser(description="Track hip X per frame with MediaPipe, then plot (no realtime).")
    parser.add_argument("--video", type=str, default="", help="Path to input video (prompted if empty).")
    parser.add_argument("--save_csv", type=str, default="", help="CSV path to save [frame, time_s, hip_x_px].")
    parser.add_argument("--save_png", type=str, default="", help="Optional PNG path to save the figure.")
    parser.add_argument("--model_complexity", type=int, default=1, help="0, 1, or 2 (default 1).")
    parser.add_argument("--min_det", type=float, default=0.5, help="Min detection confidence.")
    parser.add_argument("--min_trk", type=float, default=0.5, help="Min tracking confidence.")
    parser.add_argument("--interp", action="store_true", help="Linearly interpolate NaNs for plotting/CSV.")
    parser.add_argument("--dropna", action="store_true", help="Drop NaNs when plotting/CSV.")
    args = parser.parse_args()

    video_path = args.video.strip() or input("Enter path to your video file: ").strip()
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # FPS for time column
    _cap = cv2.VideoCapture(video_path)
    fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    _cap.release()

    # Decide a guaranteed-writable CSV path
    csv_path = _resolve_csv_path(video_path, args.save_csv)

    hip_x, width = track_hip_x(
        video_path, min_det=args.min_det, min_trk=args.min_trk, model_complexity=args.model_complexity
    )
    frames = np.arange(len(hip_x), dtype=int)
    times_s = frames / float(fps)

    # Prepare data for plotting/CSV
    x_plot = hip_x.copy()
    if args.interp and np.any(np.isfinite(x_plot)):
        nans = np.isnan(x_plot)
        x_plot[nans] = np.interp(frames[nans], frames[~nans], x_plot[~nans])

    mask = np.isfinite(x_plot) if args.dropna else np.ones_like(x_plot, dtype=bool)

    # Write CSV (with robust error reporting)
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "time_s", "hip_x_px"])
            if args.dropna:
                for fi, ti, xv in zip(frames[mask], times_s[mask], x_plot[mask]):
                    w.writerow([int(fi), float(ti), float(xv)])
            else:
                for fi, ti, xv in zip(frames, times_s, x_plot):
                    w.writerow([int(fi), float(ti), ("" if not np.isfinite(xv) else float(xv))])
        print(f"[SAVED] CSV -> {csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV to '{csv_path}': {e}")

    # ---- Plot AFTER processing (unchanged layout) ----
    plt.figure(figsize=(7, 6))
    plt.plot(x_plot[mask], frames[mask], linewidth=1.3)
    plt.xlabel("Hip X coordinate (pixels)")
    plt.ylabel("Frame #")
    plt.title("Hip X vs Frame (post-process)")
    plt.xlim(0, max(1, width))
    plt.grid(True, linestyle="--", alpha=0.4)

    if args.save_png:
        try:
            plt.savefig(args.save_png, dpi=150, bbox_inches="tight")
            print(f"[SAVED] Plot -> {os.path.abspath(args.save_png)}")
        except Exception as e:
            print(f"[ERROR] Failed to save PNG '{args.save_png}': {e}")

    plt.show()

if __name__ == "__main__":
    main()
