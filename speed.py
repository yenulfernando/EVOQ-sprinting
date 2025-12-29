# speed.py
# pip install ultralytics opencv-python-headless pandas google-cloud-storage google-cloud-firestore

import os
import cv2
import uuid
import tempfile
import pandas as pd
from datetime import datetime, timezone

from ultralytics import YOLO
from google.cloud import storage
from google.cloud import firestore

from tracker import Tracker

class PipelineError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# ================= USER SETTINGS =================
MODEL_PATH_PERSON = "models/yolov8x.pt"
MODEL_PATH_CONES  = "models/best.pt"

RUN_DIRECTION = "R2L"                # or "L2R"
START_FINISH_GAP_M = 10.0
CONF_THR = 0.25
IOU_THR  = 0.5
RESIZE_W, RESIZE_H = 1020, 500
# =================================================

# Firestore settings
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "sprint_results")

# Load models once (best practice for Cloud Run)
cone_model = YOLO(MODEL_PATH_CONES)
person_model = YOLO(MODEL_PATH_PERSON)

# Create clients once
storage_client = storage.Client()
firestore_client = firestore.Client()

def validate_video_format(path: str):
    allowed_ext = (".mp4", ".mov", ".avi", ".mkv")
    if not path.lower().endswith(allowed_ext):
        raise PipelineError(
            "Unsupported video format, please upload correct video format"
        )



# ================= FIRESTORE HELPERS =================
def write_result_to_firestore(doc_id: str, payload: dict):
    """
    Writes payload into Firestore collection.
    doc_id: document id
    payload: dict to store
    """
    firestore_client.collection(FIRESTORE_COLLECTION).document(doc_id).set(payload, merge=True)


# ================= GCS HELPERS =================
def download_from_gcs(gcs_uri: str) -> str:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS path. Must start with gs://")

    path = gcs_uri[5:]
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError("GCS path must be like gs://bucket/object")

    bucket_name, blob_path = parts

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists(storage_client):
        raise FileNotFoundError(f"GCS object not found: {gcs_uri}")

    local_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="/tmp").name
    blob.download_to_filename(local_path)
    return local_path


def upload_to_gcs(local_path: str, bucket_name: str, dest_blob: str) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


# ================= CONE DETECTION =================
def detect_cones(frame):
    res = cone_model.predict(frame, conf=0.35, verbose=False, device="cpu")[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    cones = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cones.append((cx, cy))
    return cones


def pick_track_cones(cones):
    if len(cones) < 4:
        raise PipelineError(
            "Not enough cones detected, please check cones placement and retry again"
        )
    return sorted(cones, key=lambda p: p[1])[:6]


def compute_left_right_from_cones(cones):
    cones = sorted(cones, key=lambda p: p[0])

    cL1, cL2 = cones[0], cones[1]
    cR1, cR2 = cones[-2], cones[-1]

    left_x  = int((cL1[0] + cL2[0]) / 2)
    right_x = int((cR1[0] + cR2[0]) / 2)
    return left_x, right_x


def center_of_box(x1, y1, x2, y2):
    return int((x1 + x2) // 2), int((y1 + y2) // 2)


# ================= MAIN PIPELINE =================
def run_speed(video_input: str, host_url: str = ""):
    output_bucket = os.environ.get("OUTPUT_BUCKET")
    if not output_bucket:
        raise RuntimeError("Missing OUTPUT_BUCKET environment variable")

    # Create a stable job/document id
    job_id = f"job_{uuid.uuid4().hex}"

    started_at = datetime.now(timezone.utc)

    # Write initial Firestore doc (processing)
    write_result_to_firestore(job_id, {
        "job_id": job_id,
        "status": "processing",
        "input_video": video_input,
        "run_direction": RUN_DIRECTION,
        "gap_m": START_FINISH_GAP_M,
        "created_at": started_at,
        "updated_at": started_at,
    })

    try:
        # 1) Load input video to local path
        if video_input.startswith("gs://"):
            local_video_path = download_from_gcs(video_input)
            should_cleanup_input = True
        else:
            local_video_path = video_input
            should_cleanup_input = False
        
        validate_video_format(local_video_path)

        # 2) Prepare local outputs
        output_video_path = f"/tmp/sprinter_output_{job_id}.mp4"
        output_csv_path   = f"/tmp/sprinter_splits_{job_id}.csv"

        cap = cv2.VideoCapture(local_video_path)
        if not cap.isOpened():
            raise PipelineError(
                "Invalid video input, please upload a valid sprint video"
            )

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        calc_fps = raw_fps if raw_fps and raw_fps > 1 else 120.0
        writer_fps = 120.0

        # ---------- FIRST FRAME ----------
        ok, first = cap.read()
        if not ok:
            cap.release()
            raise PipelineError(
                "Invalid video input, please upload a valid sprint video"
            )

        orig_h, orig_w = first.shape[:2]

        cones = pick_track_cones(detect_cones(first.copy()))
        left_line_orig, right_line_orig = compute_left_right_from_cones(cones)

        if RUN_DIRECTION == "L2R":
            start_line_orig, finish_line_orig = left_line_orig, right_line_orig
        else:
            start_line_orig, finish_line_orig = right_line_orig, left_line_orig

        scale_x = RESIZE_W / orig_w
        start_line = int(start_line_orig * scale_x)
        finish_line = int(finish_line_orig * scale_x)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ---------- Writer ----------
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, writer_fps, (RESIZE_W, RESIZE_H))
        if not writer.isOpened():
            raise RuntimeError("VideoWriter failed to open")

        # ---------- Tracking ----------
        tracker = Tracker()
        start_frame_by_id = {}
        counted_forward = set()
        last_x = {}
        rows = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

            res = person_model.predict(frame, conf=CONF_THR, iou=IOU_THR, verbose=False, device="cpu")[0]
            dets = []
            if res.boxes is not None:
                for box, cls_id in zip(res.boxes.xyxy.cpu().numpy(),
                                       res.boxes.cls.cpu().numpy().astype(int)):
                    if cls_id == 0:
                        x1, y1, x2, y2 = map(int, box[:4])
                        dets.append([x1, y1, x2, y2])

            tracks = tracker.update(dets)

            cv2.line(frame, (start_line, 0), (start_line, RESIZE_H), (0, 0, 255), 2)
            cv2.putText(frame, "START", (start_line - 60, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.line(frame, (finish_line, 0), (finish_line, RESIZE_H), (255, 0, 0), 2)
            cv2.putText(frame, "FINISH", (finish_line + 5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            for x1, y1, x2, y2, tid in tracks:
                cx, cy = center_of_box(x1, y1, x2, y2)

                moving_left = False
                if tid in last_x:
                    moving_left = (cx - last_x[tid]) < 0
                last_x[tid] = cx

                forward = moving_left if RUN_DIRECTION == "R2L" else (not moving_left)

                start_hit  = (start_line  < cx + 6) and (start_line  > cx - 6)
                finish_hit = (finish_line < cx + 6) and (finish_line > cx - 6)

                if start_hit and forward and tid not in start_frame_by_id:
                    start_frame_by_id[tid] = frame_idx

                if tid in start_frame_by_id and finish_hit and tid not in counted_forward:
                    frames = frame_idx - start_frame_by_id[tid]
                    if frames > 0:
                        elapsed = frames / calc_fps
                        v_ms = START_FINISH_GAP_M / elapsed
                        v_kmh = v_ms * 3.6
                        counted_forward.add(tid)

                        rows.append({
                            "id": int(tid),
                            "direction": RUN_DIRECTION,
                            "start_frame": int(start_frame_by_id[tid]),
                            "end_frame": int(frame_idx),
                            "frames": int(frames),
                            "time_s": float(elapsed),
                            "speed_m_s": float(v_ms),
                            "speed_km_h": float(v_kmh)
                        })

                        cv2.putText(frame, f"{v_kmh:.2f} km/h", (x2, y2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"FPS: {calc_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Splits: {len(rows)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)

        cap.release()
        writer.release()

        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)

        video_blob = f"outputs/{os.path.basename(output_video_path)}"
        csv_blob   = f"outputs/{os.path.basename(output_csv_path)}"

        video_url = upload_to_gcs(output_video_path, output_bucket, video_blob)
        csv_url   = upload_to_gcs(output_csv_path, output_bucket, csv_blob)

        finished_at = datetime.now(timezone.utc)

        # Write final Firestore doc (completed)
        write_result_to_firestore(job_id, {
            "status": "success" if rows else "no_splits",
            "output_bucket": output_bucket,
            "output_video_url": video_url,
            "output_csv_url": csv_url,
            "splits": rows,
            "raw_fps": float(raw_fps) if raw_fps else None,
            "calc_fps": float(calc_fps),
            "writer_fps": float(writer_fps),
            "updated_at": finished_at,
            "completed_at": finished_at,
        })

        # Cleanup
        for p in (output_video_path, output_csv_path):
            try:
                os.remove(p)
            except:
                pass
        if should_cleanup_input:
            try:
                os.remove(local_video_path)
            except:
                pass

        return {
            "job_id": job_id,
            "status": "success" if rows else "no_splits",
            "video_url": video_url,
            "csv_url": csv_url,
            "splits": rows
        }

    except PipelineError as e:
        now = datetime.now(timezone.utc)
        write_result_to_firestore(job_id, {
            "status": "failed",
            "error": e.message,
            "updated_at": now
        })
        raise

    except Exception:
        now = datetime.now(timezone.utc)
        write_result_to_firestore(job_id, {
            "status": "failed",
            "error": "Internal processing error",
            "updated_at": now
        })
        raise

