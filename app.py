# app.py
import os
from flask import Flask, request, jsonify
from speed import run_speed

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run-speed")
def run_speed_api():
    data = request.get_json(silent=True) or {}

    gcs_path = data.get("gcs_path")
    if not gcs_path:
        return jsonify({"error": "gcs_path is required"}), 400

    try:
        result = run_speed(video_input=gcs_path, host_url=request.host_url)
        return jsonify(result), 200
    except Exception as e:
        if hasattr(e, "message"):
            return jsonify({
                "status": "failed",
                "error": e.message
            }), 400

        return jsonify({
            "status": "failed",
            "error": "Internal server error"
        }), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))  # Cloud Run uses PORT=8080
    app.run(host="0.0.0.0", port=port, threaded=False)

