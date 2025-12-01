import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import argparse

def _pair_mid_y(lmk, idxL, idxR, height, min_vis=0.4):
    """Return midpoint Y (px) of a left/right landmark pair if both visible; else NaN."""
    L = lmk[idxL]; R = lmk[idxR]
    if getattr(L, "visibility", 1.0) > min_vis and getattr(R, "visibility", 1.0) > min_vis:
        return ((L.y + R.y) * 0.5) * height
    return np.nan

def track_points_y(video_path, min_det=0.5, min_trk=0.5, model_complexity=1):
    """
    Track vertical (Y) positions in pixels for:
    - mid-hip (avg of L/R hips)
    - left knee
    - left elbow
    - left shoulder
    - left toe (LEFT_FOOT_INDEX)
    Returns: (dict with arrays, width_px, height_px)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_trk
    )

    midhip_y = []
    lknee_y  = []
    lelbow_y = []
    lshould_y= []
    ltoe_y   = []

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = pose.process(frame_rgb)

            y_midhip = y_lknee = y_lelbow = y_lshould = y_ltoe = np.nan
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # mid-hip (average of L/R)
                y_midhip = _pair_mid_y(lm, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, height)
                # single left-side joints
                Lk = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                Le = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
                Ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                Lt = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]  # big toe / foot index
                if getattr(Lk, "visibility", 1.0) > 0.4:
                    y_lknee = Lk.y * height
                if getattr(Le, "visibility", 1.0) > 0.4:
                    y_lelbow = Le.y * height
                if getattr(Ls, "visibility", 1.0) > 0.4:
                    y_lshould = Ls.y * height
                if getattr(Lt, "visibility", 1.0) > 0.4:
                    y_ltoe = Lt.y * height

            midhip_y.append(float(y_midhip))
            lknee_y.append(float(y_lknee))
            lelbow_y.append(float(y_lelbow))
            lshould_y.append(float(y_lshould))
            ltoe_y.append(float(y_ltoe))
    finally:
        cap.release()
        pose.close()

    return {
        "midhip": np.array(midhip_y, dtype=float),
        "left_knee": np.array(lknee_y, dtype=float),
        "left_elbow": np.array(lelbow_y, dtype=float),
        "left_shoulder": np.array(lshould_y, dtype=float),
        "left_toe": np.array(ltoe_y, dtype=float),
    }, width, height

def _resolve_base_path(video_path, requested_base_path):
    """
    Decide a safe output *base path* (no extension).
    If requested_base_path is given, use its directory+stem.
    Otherwise default to video directory with video name as stem.
    Falls back to CWD if directory not writable.
    """
    if requested_base_path:
        base_no_ext = os.path.splitext(requested_base_path)[0]
        out_dir = os.path.dirname(base_no_ext) or "."
        stem = os.path.basename(base_no_ext)
    else:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.dirname(video_path) or "."

    try:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.abspath(os.path.join(out_dir, stem))
    except Exception:
        print(f"[WARN] Could not create/write to '{out_dir}'. Falling back to CWD.")
        return os.path.abspath(os.path.join(os.getcwd(), stem))

def _mk_outpath(base_without_ext, suffix, ext):
    """Create full output path from a base (no ext), adding suffix and extension."""
    return f"{base_without_ext}{suffix}.{ext}"

def main():
    parser = argparse.ArgumentParser(description="Track vertical motion of mid-hip, left knee/elbow/shoulder/toe; plot & CSV.")
    parser.add_argument("--video", type=str, default="", help="Path to input video (prompted if empty).")
    parser.add_argument("--save_csv", type=str, default="", help="CSV *base* path; creates *_midhip.csv, *_left_knee.csv, *_left_elbow.csv, *_left_shoulder.csv, *_left_toe.csv")
    parser.add_argument("--save_png", type=str, default="", help="PNG *base* path; creates corresponding plots.")
    parser.add_argument("--model_complexity", type=int, default=1, help="0, 1, or 2 (default 1).")
    parser.add_argument("--min_det", type=float, default=0.5, help="Min detection confidence.")
    parser.add_argument("--min_trk", type=float, default=0.5, help="Min tracking confidence.")
    parser.add_argument("--interp", action="store_true", help="Linearly interpolate NaNs for plotting/CSV.")
    parser.add_argument("--dropna", action="store_true", help="Drop NaNs when plotting/CSV.")
    args = parser.parse_args()

    video_path = args.video.strip() or input("Enter path to your video file: ").strip()
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # FPS (for time column)
    _cap = cv2.VideoCapture(video_path)
    fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    _cap.release()

    # Resolve base paths (no extension)
    csv_base = _resolve_base_path(video_path, args.save_csv)
    png_base = _resolve_base_path(video_path, args.save_png) if args.save_png else None

    data_dict, width, height = track_points_y(
        video_path, min_det=args.min_det, min_trk=args.min_trk, model_complexity=args.model_complexity
    )
    n_frames = len(next(iter(data_dict.values())))
    frames = np.arange(n_frames, dtype=int)
    times_s = frames / float(fps)

    # Prepare (optionally interpolate / dropna) per series
    prepared = {}
    masks = {}
    for key, arr in data_dict.items():
        y_plot = arr.copy()
        if args.interp and np.any(np.isfinite(y_plot)):
            nans = np.isnan(y_plot)
            y_plot[nans] = np.interp(frames[nans], frames[~nans], y_plot[~nans])
        mask = np.isfinite(y_plot) if args.dropna else np.ones_like(y_plot, dtype=bool)
        prepared[key] = y_plot
        masks[key] = mask

    # --- Write 5 CSVs (frame_index, time_s, <joint>_y_px) ---
    csv_names = {
        "midhip":        _mk_outpath(csv_base, "_midhip", "csv"),
        "left_knee":     _mk_outpath(csv_base, "_left_knee", "csv"),
        "left_elbow":    _mk_outpath(csv_base, "_left_elbow", "csv"),
        "left_shoulder": _mk_outpath(csv_base, "_left_shoulder", "csv"),
        "left_toe":      _mk_outpath(csv_base, "_left_toe", "csv"),
    }
    for key in ["midhip", "left_knee", "left_elbow", "left_shoulder", "left_toe"]:
        path = csv_names[key]
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_index", "time_s", f"{key}_y_px"])
                if args.dropna:
                    for fi, ti, yv in zip(frames[masks[key]], times_s[masks[key]], prepared[key][masks[key]]):
                        w.writerow([int(fi), float(ti), float(yv)])
                else:
                    for fi, ti, yv in zip(frames, times_s, prepared[key]):
                        w.writerow([int(fi), float(ti), ("" if not np.isfinite(yv) else float(yv))])
            print(f"[SAVED] CSV -> {os.path.abspath(path)}")
        except Exception as e:
            print(f"[ERROR] Failed to write CSV '{path}': {e}")

    # --- Plot 5 graphs (x=frame index, y=vertical position in px) ---
    titles = {
        "midhip":        "Mid-Hip Y vs Frame (0=top)",
        "left_knee":     "Left Knee Y vs Frame (0=top)",
        "left_elbow":    "Left Elbow Y vs Frame (0=top)",
        "left_shoulder": "Left Shoulder Y vs Frame (0=top)",
        "left_toe":      "Left Toe (Foot Index) Y vs Frame (0=top)",
    }
    png_paths = {
        "midhip":        (_mk_outpath(png_base, "_midhip", "png") if png_base else None),
        "left_knee":     (_mk_outpath(png_base, "_left_knee", "png") if png_base else None),
        "left_elbow":    (_mk_outpath(png_base, "_left_elbow", "png") if png_base else None),
        "left_shoulder": (_mk_outpath(png_base, "_left_shoulder", "png") if png_base else None),
        "left_toe":      (_mk_outpath(png_base, "_left_toe", "png") if png_base else None),
    }

    for key in ["midhip", "left_knee", "left_elbow", "left_shoulder", "left_toe"]:
        plt.figure(figsize=(7, 6))
        plt.plot(frames[masks[key]], prepared[key][masks[key]], linewidth=1.3)
        plt.xlabel("Frame #")
        plt.ylabel("Vertical position (pixels, 0=top)")
        plt.title(titles[key])
        plt.ylim(0, max(1, height))   # image coordinates: 0 top, height bottom
        plt.grid(True, linestyle="--", alpha=0.4)
        if png_paths[key]:
            try:
                plt.savefig(png_paths[key], dpi=150, bbox_inches="tight")
                print(f"[SAVED] Plot -> {os.path.abspath(png_paths[key])}")
            except Exception as e:
                print(f"[ERROR] Failed to save PNG '{png_paths[key]}': {e}")

    plt.show()

if __name__ == "__main__":
    main()
