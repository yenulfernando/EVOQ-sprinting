# analyze_quickrun.py
# QuickRun: edit CONFIG below, then Run in PyCharm.

import csv, os, math
import numpy as np

# ========= CONFIG (EDIT THESE) =========
CSV_PATH = r"E:\Evoq\Sprinting video\useful\IMG_0772_hip_x.csv"  # e.g. r"D:\runs\IMG_0772_hip_x.csv". If empty, script will ask.
DISTANCE_M = 5.0           # real distance between start & finish (meters)

# Choose one way to mark the segment:
# A) by pixel columns (recommended if you know start/finish lines in image):
START_X_PX  = None          # e.g. 60.0
FINISH_X_PX = None          # e.g. 1280.0

# B) by frames (0-based frame indices):
START_FRAME  = 127        # e.g. 15
FINISH_FRAME = 160         # e.g. 145

# If you already know meters-per-pixel along run direction, put it here.
# If left None and you provided START_X_PX/FINISH_X_PX, scale will be derived from DISTANCE_M / |dx_px|.
SCALE_M_PER_PX = None

SMOOTH_WIN = 5              # moving-average window for smoothing (odd int)
SPLITS_M = [5, 10]          # split distances to report (meters); leave [] to skip
SAVE_METRICS_CSV = ""       # optional: path to save time_s,dist_m,speed,accel
# =======================================

def load_csv(path):
    frames, times, xs = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            frames.append(int(row["frame_index"]))
            times.append(float(row["time_s"]))
            x = row["hip_x_px"]
            xs.append(float(x) if x not in ("", None) else np.nan)
    F, T, X = np.array(frames), np.array(times, float), np.array(xs, float)
    # fill small gaps
    good = np.isfinite(X)
    if np.any(good) and np.any(~good):
        X[~good] = np.interp(F[~good], F[good], X[good])
    return F, T, X

def moving_average(x, k=5):
    if k <= 1: return x
    k = int(k)
    pad = k//2
    xpad = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k)/k
    return np.convolve(xpad, ker, mode="valid")

def crossing_index(x, target):
    for i in range(len(x)-1):
        a, b = x[i], x[i+1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if (a - target) == 0:
            return i
        if (a - target) * (b - target) < 0:
            return i+1
    return None

def main():
    global CSV_PATH
    if not CSV_PATH:
        CSV_PATH = input("Enter path to your hip_x CSV: ").strip()
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    F, T, Xpx = load_csv(CSV_PATH)
    dt = np.diff(T)
    fps_est = 1.0 / np.median(dt[dt > 0]) if np.any(dt > 0) else float("nan")
    print(f"[INFO] Samples: {len(F)} | FPS (from CSV): {fps_est:.3f}")

    # ---- Average speed (timing only) ----
    t_start = t_finish = None
    if START_FRAME is not None and FINISH_FRAME is not None:
        if FINISH_FRAME <= START_FRAME:
            print("[WARN] FINISH_FRAME must be > START_FRAME.")
        else:
            t_start, t_finish = T[START_FRAME], T[FINISH_FRAME]
            seg_time = (t_finish - t_start) / 2
            v_avg = DISTANCE_M / seg_time
            print(f"[AVG SPEED] {v_avg:.3f} m/s  ({v_avg*3.6:.2f} km/h) over {DISTANCE_M} m in {seg_time:.3f} s "
                  f"(frames {START_FRAME}→{FINISH_FRAME})")

    elif START_X_PX is not None and FINISH_X_PX is not None:
        i_start = crossing_index(Xpx, START_X_PX)
        i_finish = crossing_index(Xpx, FINISH_X_PX)
        if i_start is None or i_finish is None or i_finish <= i_start:
            print("[WARN] Could not find valid start/finish crossings in CSV.")
        else:
            t_start, t_finish = T[i_start], T[i_finish]
            seg_time = t_finish - t_start
            v_avg = DISTANCE_M / seg_time
            print(f"[AVG SPEED] {v_avg:.3f} m/s  ({v_avg*3.6:.2f} km/h) over {DISTANCE_M} m in {seg_time:.3f} s "
                  f"(frames {F[i_start]}→{F[i_finish]})")

            # derive scale if not provided
            dx_px = abs(FINISH_X_PX - START_X_PX)
            if dx_px > 0 and SCALE_M_PER_PX is None:
                scale = DISTANCE_M / dx_px
                print(f"[INFO] Derived scale from segment: {scale:.6f} m/px")
                scale_to_use = scale
            else:
                scale_to_use = SCALE_M_PER_PX
    else:
        scale_to_use = SCALE_M_PER_PX
        print("[NOTE] Set either START/FINISH frames or pixel X to compute average speed.")

    # ---- Instantaneous speed/accel + splits (needs scale) ----
    if 'scale_to_use' not in locals():
        scale_to_use = SCALE_M_PER_PX

    if scale_to_use:
        # choose segment [start:finish] if known, else whole trace
        if t_start is not None and t_finish is not None:
            mask = (T >= t_start) & (T <= t_finish)
        else:
            mask = np.ones_like(T, dtype=bool)

        t_seg = T[mask]
        x_m = scale_to_use * Xpx[mask]
        x_sm = moving_average(x_m, k=SMOOTH_WIN)
        v = np.gradient(x_sm, t_seg)          # m/s
        a = np.gradient(v, t_seg)             # m/s^2

        print(f"[PEAK] speed: {np.nanmax(v):.3f} m/s ({np.nanmax(v)*3.6:.2f} km/h)")
        print(f"[MEAN] speed (segment): {np.nanmean(v):.3f} m/s")
        print(f"[ACCEL] max: {np.nanmax(a):.3f} m/s² | min: {np.nanmin(a):.3f} m/s²")

        # Splits
        if SPLITS_M:
            d = x_sm - x_sm[0]
            for dm in SPLITS_M:
                idxs = np.where(d >= dm)[0]
                if len(idxs) == 0:
                    print(f"[SPLIT] {dm:.1f} m not reached.")
                else:
                    print(f"[SPLIT] {dm:.1f} m in {t_seg[idxs[0]] - t_seg[0]:.3f} s")

        # Optional: save time, dist, speed, accel
        if SAVE_METRICS_CSV:
            with open(SAVE_METRICS_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "dist_m", "speed_mps", "accel_mps2"])
                d = x_sm - x_sm[0]
                for ti, di, vi, ai in zip(t_seg, d, v, a):
                    w.writerow([float(ti), float(di), float(vi), float(ai)])
            print(f"[SAVED] metrics -> {os.path.abspath(SAVE_METRICS_CSV)}")
    else:
        print("[NOTE] Provide SCALE_M_PER_PX (or START/FINISH pixel X so the script can derive it) "
              "to get instantaneous speed, acceleration, and splits.")

if __name__ == "__main__":
    main()
