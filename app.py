import os
import uuid
import threading
import time
import json
from typing import Dict, Any, List

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash

import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "web_outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
WEIGHTS_PATH = os.path.join(BASE_DIR, "mobilenetv2_violence.weights.h5")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", "dev-secret-change-me")
login_manager = LoginManager(app)
login_manager.login_view = "login"
DEFAULT_THRESHOLD = 0.5
current_threshold = DEFAULT_THRESHOLD


# In-memory job store
jobs_lock = threading.Lock()
jobs: Dict[str, Dict[str, Any]] = {}
recent_job_ids: List[str] = []

_model: Model = None
_model_lock = threading.Lock()


# Simple single-user auth (Controller)
class Controller(UserMixin):
    id = "controller"


_controller_user = Controller()
_controller_username = os.environ.get("CONTROLLER_USERNAME", "admin")
_controller_password_hash = os.environ.get("CONTROLLER_PASSWORD_HASH")
if not _controller_password_hash:
    _controller_password_hash = generate_password_hash(os.environ.get("CONTROLLER_PASSWORD", "admin123"))


@login_manager.user_loader
def load_user(user_id):
    if user_id == _controller_user.id:
        return _controller_user
    return None


def get_model() -> Model:
    global _model
    with _model_lock:
        if _model is None:
            base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            output = Dense(1, activation='sigmoid')(x)
            _model = Model(inputs=base_model.input, outputs=output)
            if not os.path.exists(WEIGHTS_PATH):
                raise RuntimeError("Model weights not found: " + WEIGHTS_PATH)
            _model.load_weights(WEIGHTS_PATH)
        return _model


def summarize_segments(frame_scores: List[float], fps: float, threshold: float = 0.5, min_duration_sec: float = 0.5) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    start_idx = None
    for idx, score in enumerate(frame_scores):
        if score >= threshold and start_idx is None:
            start_idx = idx
        elif score < threshold and start_idx is not None:
            end_idx = idx - 1
            duration = (end_idx - start_idx + 1) / fps if fps else 0
            if duration >= min_duration_sec:
                segments.append({
                    "start": start_idx / fps if fps else 0,
                    "end": end_idx / fps if fps else 0,
                    "avg_score": float(np.mean(frame_scores[start_idx:end_idx + 1]))
                })
            start_idx = None
    if start_idx is not None:
        end_idx = len(frame_scores) - 1
        duration = (end_idx - start_idx + 1) / fps if fps else 0
        if duration >= min_duration_sec:
            segments.append({
                "start": start_idx / fps if fps else 0,
                "end": end_idx / fps if fps else 0,
                "avg_score": float(np.mean(frame_scores[start_idx:end_idx + 1]))
            })
    return segments


def write_csv_report(csv_path: str, segments: List[Dict[str, Any]], total_duration: float, violence_ratio: float) -> None:
    import csv
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Start (s)", "End (s)", "Avg Score (0-1)"])
        for seg in segments:
            writer.writerow([f"{seg['start']:.2f}", f"{seg['end']:.2f}", f"{seg['avg_score']:.4f}"])
        writer.writerow([])
        writer.writerow(["Total Duration (s)", f"{total_duration:.2f}"])
        writer.writerow(["Violence Duration (s)", f"{sum(seg['end']-seg['start'] for seg in segments):.2f}"])
        writer.writerow(["Violence Ratio", f"{violence_ratio:.3f}"])


def process_video_background(job_id: str, input_path: str, output_path: str, report_json_path: str, report_csv_path: str) -> None:
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "loading_model"
            jobs[job_id]["message"] = "Loading model"

        model = get_model()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        frame_scores: List[float] = []
        snapshot_saved = False

        with jobs_lock:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["message"] = "Analyzing"
            jobs[job_id]["progress"] = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (224, 224)).astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            pred = float(model.predict(img, verbose=0)[0][0])
            frame_scores.append(pred)
            is_violence = pred > jobs[job_id].get("threshold", current_threshold)
            color = (0, 0, 255) if is_violence else (0, 255, 0)
            label = f"{'VIOLENCE' if is_violence else 'SAFE'} {pred:.2f}"
            cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            writer.write(frame)

            # Save a snapshot for history (first written frame)
            if not snapshot_saved:
                snap_path = os.path.join(UPLOAD_DIR, f"{job_id}.jpg")
                try:
                    cv2.imwrite(snap_path, frame)
                    snapshot_saved = True
                    with jobs_lock:
                        jobs[job_id]["snapshot"] = os.path.basename(snap_path)
                except Exception:
                    pass

            frame_idx += 1
            if total_frames:
                progress = int((frame_idx / total_frames) * 100)
            else:
                progress = 0

            if frame_idx % 10 == 0 or frame_idx == total_frames:
                with jobs_lock:
                    jobs[job_id]["progress"] = progress
                    jobs[job_id]["message"] = "Violence Detected" if is_violence else "Analyzing"

        writer.release()
        cap.release()

        segments = summarize_segments(frame_scores, fps, threshold=jobs[job_id].get("threshold", current_threshold))
        violence_frames = sum(int(round((seg["end"] - seg["start"]) * fps)) for seg in segments)
        total_duration = (total_frames / fps) if fps else 0
        violence_ratio = (violence_frames / total_frames) if total_frames else 0.0
        max_score = float(max(frame_scores) if frame_scores else 0.0)
        final_threshold = jobs[job_id].get("threshold", current_threshold)
        is_violent_overall = (len(segments) > 0) or (max_score > final_threshold)
        final_confidence = max_score if is_violent_overall else (1.0 - max_score)

        report_data = {
            "segments": segments,
            "fps": fps,
            "total_frames": total_frames,
            "total_duration": total_duration,
            "violence_ratio": violence_ratio,
            "output_video": os.path.basename(output_path),
            "final_prediction": "VIOLENT" if is_violent_overall else "SAFE",
            "final_confidence": final_confidence,
        }

        with open(report_json_path, "w") as f:
            json.dump(report_data, f, indent=2)

        write_csv_report(report_csv_path, segments, total_duration, violence_ratio)

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Completed"
            jobs[job_id]["result"] = report_data
            # maintain history (most recent first, cap 50)
            recent_job_ids.insert(0, job_id)
            if len(recent_job_ids) > 50:
                recent_job_ids.pop()
            jobs[job_id]["ended_at"] = time.time()
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = str(e)


@app.route("/")
@login_required
def index():
    return redirect(url_for("dashboard"))


ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "video" not in request.files:
        abort(400, description="No file part")
    file = request.files["video"]
    if file.filename == "":
        abort(400, description="No selected file")
    if not allowed_file(file.filename):
        abort(400, description="Unsupported file type")

    job_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    report_json_path = os.path.join(REPORT_DIR, f"{job_id}.json")
    report_csv_path = os.path.join(REPORT_DIR, f"{job_id}.csv")
    file.save(input_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "message": "Queued",
            "progress": 0,
            "input": os.path.basename(input_path),
            "output": os.path.basename(output_path),
            "threshold": current_threshold,
            "started_at": time.time(),
        }

    t = threading.Thread(target=process_video_background, args=(job_id, input_path, output_path, report_json_path, report_csv_path), daemon=True)
    t.start()

    return redirect(url_for("status_page", job_id=job_id))


@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "no_file"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "empty_filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "unsupported_type"}), 400

    job_id = str(uuid.uuid4())
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    report_json_path = os.path.join(REPORT_DIR, f"{job_id}.json")
    report_csv_path = os.path.join(REPORT_DIR, f"{job_id}.csv")
    file.save(input_path)

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "message": "Queued",
            "progress": 0,
            "input": os.path.basename(input_path),
            "output": os.path.basename(output_path),
            "threshold": current_threshold,
            "started_at": time.time(),
        }

    t = threading.Thread(target=process_video_background, args=(job_id, input_path, output_path, report_json_path, report_csv_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
@login_required
def status_page(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        abort(404)
    return render_template("status.html", job_id=job_id)


@app.route("/api/status/<job_id>")
@login_required
def api_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "not_found"}), 404
        return jsonify(job)


@app.route("/result/<job_id>")
@login_required
def result(job_id: str):
    json_path = os.path.join(REPORT_DIR, f"{job_id}.json")
    if not os.path.exists(json_path):
        abort(404)
    with open(json_path, "r") as f:
        report = json.load(f)
    return render_template("result.html", job_id=job_id, report=report)


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    global current_threshold
    if request.method == "POST":
        try:
            val = float(request.form.get("threshold", current_threshold))
            if 0.0 <= val <= 1.0:
                current_threshold = val
        except Exception:
            pass
        return redirect(url_for("settings"))
    return render_template("settings.html", threshold=current_threshold)


@app.route("/history")
@login_required
def history():
    with jobs_lock:
        items = []
        for jid in recent_job_ids:
            j = jobs.get(jid)
            if not j:
                continue
            items.append({
                "id": jid,
                "status": j.get("status"),
                "input": j.get("input"),
                "output": j.get("output"),
                "snapshot": j.get("snapshot"),
            })
    return render_template("history.html", jobs=items)


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/dashboard")
@login_required
def dashboard():
    with jobs_lock:
        total = len([jid for jid in recent_job_ids if jobs.get(jid)])
        violent = 0
        durations = []
        recent = []
        for jid in recent_job_ids[:6]:
            j = jobs.get(jid)
            if not j:
                continue
            res = j.get("result") or {}
            if res.get("final_prediction") == "VIOLENT":
                violent += 1
            if j.get("started_at") and j.get("ended_at"):
                durations.append(j["ended_at"] - j["started_at"])
            recent.append({
                "id": jid,
                "status": j.get("status"),
                "prediction": (res.get("final_prediction") or "PENDING"),
                "output": j.get("output"),
            })
    safe = max(total - violent, 0)
    avg_response = sum(durations)/len(durations) if durations else 0.0
    return render_template("dashboard.html", stats={
        "total": total,
        "violent": violent,
        "safe": safe,
        "avg_response": avg_response,
    }, recent=recent)


@app.route("/alerts")
@login_required
def alerts():
    with jobs_lock:
        items = []
        for jid in recent_job_ids:
            j = jobs.get(jid)
            if not j:
                continue
            res = j.get("result") or {}
            if res.get("final_prediction") == "VIOLENT":
                items.append({
                    "id": jid,
                    "time": j.get("ended_at") or j.get("started_at"),
                    "output": j.get("output"),
                })
    return render_template("alerts.html", alerts_list=items)


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user={
        "username": _controller_username,
        "role": "Security Operator",
    })


@app.route("/download/video/<job_id>")
@login_required
def download_video(job_id: str):
    file_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(OUTPUT_DIR, f"{job_id}.mp4", as_attachment=True)


@app.route("/download/report/<job_id>.csv")
@login_required
def download_csv(job_id: str):
    file_path = os.path.join(REPORT_DIR, f"{job_id}.csv")
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(REPORT_DIR, f"{job_id}.csv", as_attachment=True)


@app.route("/uploads/<path:filename>")
@login_required
def serve_upload(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<path:filename>")
@login_required
def serve_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == _controller_username and check_password_hash(_controller_password_hash, password):
            login_user(_controller_user)
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid credentials"), 401
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


