import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, send_file
import tempfile
import shutil
from google.cloud import storage
from pose_pipeline import run_full_pipeline
import uuid
from datetime import datetime, timezone
from google.cloud import firestore
import csv
import base64
import zipfile
from pose_pipeline import PipelineError


def csv_and_images_to_firestore_json(csv_path, zip_path):
    results = []

    # Open ZIP once
    with zipfile.ZipFile(zip_path, "r") as zf:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_path = row.get("IMAGE_PATH", "")
                image_name = os.path.basename(image_path)

                image_b64 = None
                if image_name in zf.namelist():
                    image_bytes = zf.read(image_name)
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                # Separate angles cleanly
                angles = {
                    k: float(v) if v not in ("", None) else None
                    for k, v in row.items()
                    if k not in ("POSE", "FRAME", "CONFIDENCE", "IMAGE_PATH")
                }

                results.append({
                    "POSE": row["POSE"],
                    "FRAME": int(row["FRAME"]),
                    "CONFIDENCE": float(row["CONFIDENCE"]),
                    "angles": angles,
                    "image_base64": image_b64
                })

    return results


app = Flask(__name__)

FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "pose_results")
firestore_client = firestore.Client()

def write_pose_result(job_id: str, payload: dict):
    firestore_client.collection(FIRESTORE_COLLECTION).document(job_id).set(payload, merge=True)


# -------------------------------
# Helper: download from GCS
# -------------------------------
def download_from_gcs(gcs_path):
    if not gcs_path.startswith("gs://"):
        raise ValueError("Invalid GCS path")

    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    file_path = "/".join(gcs_path.replace("gs://", "").split("/")[1:])

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    temp_local = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    blob.download_to_filename(temp_local)

    return temp_local


# -------------------------------
# ROUTE 1 — Upload File Directly
# -------------------------------
@app.route("/", methods=["POST"])
def handle_video():
    job_id = f"pose_{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc)

    write_pose_result(job_id, {
        "job_id": job_id,
        "status": "processing",
        "source": "upload",
        "created_at": now,
        "updated_at": now,
    })

    if "file" not in request.files:
        write_pose_result(job_id, {"status": "failed", "error": "Missing file", "updated_at": datetime.now(timezone.utc)})
        return jsonify({"error": "Upload file using key 'file'", "job_id": job_id}), 400

    video = request.files["file"]
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    video.save(temp_path)

    try:
        run_full_pipeline(temp_path)
        return build_response(job_id=job_id, host_url=request.host_url)
    except PipelineError as pe:
        write_pose_result(job_id, {
            "job_id": job_id,
            "status": "failed",
            "reason": pe.user_message,
            "updated_at": datetime.now(timezone.utc),
        })

        return jsonify({
            "job_id": job_id,
            "status": "failed",
            "reason": pe.user_message
        }), 400

    except Exception as e:
        write_pose_result(job_id, {
            "job_id": job_id,
            "status": "failed",
            "reason": "Internal processing error. Please try again.",
            "updated_at": datetime.now(timezone.utc),
        })

        return jsonify({
            "job_id": job_id,
            "status": "failed",
            "reason": "Internal processing error. Please try again."
        }), 500



# -------------------------------
# ROUTE 2 — Load From GCS
# -------------------------------
@app.route("/run-pose", methods=["POST"])
def handle_gcs():
    job_id = f"pose_{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc)

    data = request.get_json(silent=True) or {}
    gcs_path = data.get("gcs_path")

    write_pose_result(job_id, {
        "job_id": job_id,
        "status": "processing",
        "source": "gcs",
        "gcs_path": gcs_path,
        "created_at": now,
        "updated_at": now,
    })

    if not gcs_path:
        write_pose_result(job_id, {"status": "failed", "error": "Missing gcs_path", "updated_at": datetime.now(timezone.utc)})
        return jsonify({"error": "Send JSON: { 'gcs_path': 'gs://bucket/video.mp4' }", "job_id": job_id}), 400

    try:
        temp_path = download_from_gcs(gcs_path)
        run_full_pipeline(temp_path)
        return build_response(job_id=job_id, host_url=request.host_url)
    except Exception as e:
        write_pose_result(job_id, {"status": "failed", "error": str(e), "updated_at": datetime.now(timezone.utc)})
        return jsonify({"error": str(e), "job_id": job_id}), 500



# -------------------------------
# Build ZIP + CSV Response
# -------------------------------
def build_response(job_id: str, host_url: str):
    BASE_DIR = os.getenv("POSE_BASE_DIR", "pose_pipeline")

    frames_folder = os.path.join(BASE_DIR, "best_pose_frames_from_kalman")
    angles_file = os.path.join(BASE_DIR, "best_pose_angles_summary.csv")

    tmp_dir = tempfile.gettempdir()

    # ---- ZIP FRAMES ----
    zip_output_path = os.path.join(tmp_dir, "best_pose_frames_from_kalman.zip")
    if os.path.exists(zip_output_path):
        os.remove(zip_output_path)

    if os.path.exists(frames_folder):
        shutil.make_archive(
            base_name=zip_output_path.replace(".zip", ""),
            format="zip",
            root_dir=frames_folder
        )

    # ---- COPY CSV ----
    tmp_angles = os.path.join(tmp_dir, "best_pose_angles_summary.csv")
    if os.path.exists(angles_file):
        shutil.copy(angles_file, tmp_angles)

    # Build same output you already return (but as absolute URLs)
    frames_url = f"{host_url}download?file=best_pose_frames_from_kalman.zip"
    angles_url = f"{host_url}download?file=best_pose_angles_summary.csv"

    response_payload = {
        "job_id": job_id,
        "status": "success",
        "frames_zip": frames_url,
        "angles_csv": angles_url,
    }

    # ✅ Write final result to Firestore
    csv_path = os.path.join(tmp_dir, "best_pose_angles_summary.csv")
    zip_path = zip_output_path

    pose_json = csv_and_images_to_firestore_json(csv_path, zip_path)

    write_pose_result(job_id, {
        "job_id": job_id,
        "status": "success",
        "results": pose_json,
        "updated_at": datetime.now(timezone.utc),
    })


    #return jsonify(response_payload)

    # ---- COPY CSV ----
    tmp_angles = os.path.join(tmp_dir, "best_pose_angles_summary.csv")
    if os.path.exists(angles_file):
        shutil.copy(angles_file, tmp_angles)

    return jsonify({
        "status": "success",
        "frames_zip": "/download?file=best_pose_frames_from_kalman.zip",
        "angles_csv": "/download?file=best_pose_angles_summary.csv",
        "response_payload": response_payload
    })


# -------------------------------
# FILE DOWNLOAD ROUTE (NEEDED)
# -------------------------------
@app.route("/download")
def download_file():
    filename = request.args.get("file")
    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path, as_attachment=True)


# -------------------------------
# Cloud Run Start
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
