from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from thermal_analysis import analyze_single_image, analyze_video_temperature
from respiratory_rate import estimate_respiratory_rate
from status_flagger import evaluate_vitals_status
from rr_model import (DATASET_DIR, MODEL_DIR, list_dataset_videos, features_from_rr_result,
                      predict_ml, load_metrics)
from rr_signal_methods import compare_rr_methods
from benchmark import load_benchmark_history
import json
import os
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

def _get_extension(filename):
    """Extract lowercase file extension from filename."""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Thermal vital signs analysis service is running"}), 200

@app.route('/api/history', methods=['GET'])
def history():
    return jsonify([]), 200

@app.route('/api/analyze', methods=['POST'])
def analyze():
    return handle_analyze(request, debug=False)

@app.route('/api/analyze/debug', methods=['POST'])
def analyze_debug():
    return handle_analyze(request, debug=True)

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    """Analyze a thermal video for fever, respiratory rate (signal + ML)."""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided in request"}), 400
            
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected video file"}), 400
        
        ext = _get_extension(file.filename)
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            if ext in ALLOWED_IMAGE_EXTENSIONS:
                return jsonify({"error": f"Invalid file type: received an image file (.{ext}). Use the /api/analyze endpoint for image files, or upload a video (.mp4, .avi, .mov) to this endpoint."}), 400
            return jsonify({"error": f"Unsupported file type: .{ext}. Supported video formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"}), 400
            
        temp_filename = f"{uuid.uuid4().hex}.{ext}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)
        
        debug_flag = request.args.get('debug', 'false').lower() == 'true'
        try:
            response_data = analyze_video_file(temp_path, file.filename, debug_flag)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataset/videos', methods=['GET'])
def dataset_videos():
    """List the bundled 'Thermal Videos' dataset (with hand-counted labels where known)."""
    return jsonify(list_dataset_videos()), 200

@app.route('/api/dataset/analyze/<name>', methods=['POST', 'GET'])
def analyze_dataset_video(name):
    """Run the full fever + respiratory pipeline on one of the bundled dataset videos."""
    known = {v["filename"]: v for v in list_dataset_videos()}
    if name not in known:  # whitelist lookup also blocks path traversal
        return jsonify({"error": f"Unknown dataset video: {name}"}), 404
    try:
        debug_flag = request.args.get('debug', 'false').lower() == 'true'
        data = analyze_video_file(os.path.join(DATASET_DIR, name), name, debug_flag)
        data["dataset_label_bpm"] = known[name]["label_bpm"]
        metrics = load_metrics() or {}
        data["dataset_oof_prediction"] = (metrics.get("oof_predictions") or {}).get(name)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/metrics', methods=['GET'])
def model_metrics():
    """Training/evaluation metrics of the respiratory-rate ML models (from rr_model.py train)."""
    metrics = load_metrics()
    if metrics is None:
        return jsonify({"error": "Models not trained yet. Run: python rr_model.py train"}), 404
    return jsonify({k: v for k, v in metrics.items() if k != "oof_predictions"}), 200

@app.route('/api/benchmark/history', methods=['GET'])
def benchmark_history():
    """Retrieve complete benchmark history tracking all improvement steps."""
    return jsonify(load_benchmark_history()), 200

@app.route('/result/<path:filename>', methods=['GET'])
def serve_result_file(filename):
    """Serve confusion matrix charts and result assets."""
    return send_from_directory('result', filename)

ALLOWED_MODEL_FILES = {"tuning_mae_bar.png", "tuning_pred_vs_ref.png"}

@app.route('/models/<path:filename>', methods=['GET'])
def serve_model_file(filename):
    """Serve model report artifacts and plots (whitelisted)."""
    if filename not in ALLOWED_MODEL_FILES:
        return jsonify({"error": "Forbidden or not found"}), 404
    return send_from_directory(MODEL_DIR, filename)

@app.route('/api/model/tuning', methods=['GET'])
def model_tuning():
    """Hyperparameter tuning results from nested-CV."""
    path = os.path.join(MODEL_DIR, "tuning_results.json")
    if not os.path.exists(path):
        return jsonify({"available": False}), 200
    try:
        with open(path) as f:
            data = json.load(f)
        data["available"] = True
        report_path = os.path.join(MODEL_DIR, "tuning_report.md")
        conclusion = ""
        if os.path.exists(report_path):
            with open(report_path) as f:
                content = f.read()
            if "## Honest conclusion" in content:
                conclusion = content.split("## Honest conclusion")[1].split("![")[0].strip()
        data["honest_conclusion"] = conclusion
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"available": False, "error": str(e)}), 500

def analyze_video_file(video_path, filename, debug_flag=False):
    """Shared video pipeline: respiratory rate (signal + ML + multi-method) and video temperature/fever."""
    # 1. Respiratory rate (signal processing). Always ask for the signal: the ML features and the
    #    dashboard waveform are derived from it.
    rr_result = estimate_respiratory_rate(video_path, debug=True)
    ml_feats, rr_signal = features_from_rr_result(rr_result)

    # 2. ML respiratory rate (models from V5.ipynb, see rr_model.py)
    ml_preds = predict_ml(ml_feats)
    ml_rr = {
        "available": bool(ml_preds),
        "predictions": ml_preds,
        "ensemble_bpm": ml_preds.get("Ensemble"),
        "features": ml_feats,
    }

    # 3. New multi-method signal analysis (rr_signal_methods.py)
    #    Runs on the same already-filtered signal that rr_signal uses.
    signal_methods_result = {}
    dbg = rr_result.get("debug") or {}
    filtered_arr = dbg.get("filtered_signal")
    fps = rr_result.get("fps")
    _sig_arr = None
    if filtered_arr is not None and fps:
        import numpy as _np
        _sig_arr = _np.asarray(filtered_arr, dtype=float)
    elif rr_signal and rr_signal.get("filtered") and fps:
        import numpy as _np
        _sig_arr = _np.asarray(rr_signal["filtered"], dtype=float)

    if _sig_arr is not None and fps:
        import numpy as _np
        signal_methods_result = compare_rr_methods(_sig_arr, fps)
        # Compute consensus (median of valid BPMs + confidence from spread)
        valid_bpms = [
            r["bpm"] for r in signal_methods_result.values()
            if isinstance(r, dict) and r.get("bpm") is not None
        ]
        if valid_bpms:
            consensus_bpm = round(float(_np.median(valid_bpms)), 1)
            spread = max(valid_bpms) - min(valid_bpms)
            if len(valid_bpms) >= 3 and spread < 3.0:
                consensus_confidence = "high"
            elif len(valid_bpms) >= 2 and spread < 5.0:
                consensus_confidence = "medium"
            else:
                consensus_confidence = "low"
            signal_methods_result["consensus_bpm"] = consensus_bpm
            signal_methods_result["consensus_confidence"] = consensus_confidence
        signal_methods_result["methods_succeeded"] = len(valid_bpms)
        signal_methods_result["methods_available"] = len([
            r for r in signal_methods_result.values()
            if isinstance(r, dict) and r.get("confidence") != "unavailable"
        ])

    # 4. Video Temperature & Fever Analysis (reusing validated thermal_analysis pipeline)
    temp_result = analyze_video_temperature(video_path, num_samples=5, debug=debug_flag)

    # Synthetic test videos are labelled as such in the UI
    is_synthetic = "sample_thermal_video" in filename.lower() or "_smoke_" in filename.lower()

    # Headline respiratory rate: signal-processing estimate when it is valid, otherwise the ML ensemble
    # (unchanged — the new methods are display-only diagnostics)
    signal_rr = rr_result.get("respiratory_rate_bpm")
    computed_rr = signal_rr if signal_rr is not None else ml_rr["ensemble_bpm"]
    computed_temp = temp_result.get("forehead_temp_c") or temp_result.get("face_neck_temp_c")
    status_flags = evaluate_vitals_status(computed_temp, computed_rr)
    if not debug_flag:
        rr_result.pop("debug", None)

    return {
        "is_synthetic": is_synthetic,
        "filename": filename,
        # Temperature fields (same contract as /api/analyze)
        "face_neck_skin_temp_c": temp_result.get("face_neck_skin_temp_c"),
        "forehead_skin_temp_c": temp_result.get("forehead_skin_temp_c"),
        "inner_canthus_skin_temp_c": temp_result.get("inner_canthus_skin_temp_c"),
        "face_neck_temp_c": temp_result.get("face_neck_temp_c"),
        "face_neck_temp_f": temp_result.get("face_neck_temp_f"),
        "forehead_temp_c": temp_result.get("forehead_temp_c"),
        "forehead_temp_f": temp_result.get("forehead_temp_f"),
        "inner_canthus_temp_c": temp_result.get("inner_canthus_temp_c"),
        "inner_canthus_temp_f": temp_result.get("inner_canthus_temp_f"),
        "fever_detected": temp_result.get("fever_detected", False),
        "fever_threshold_c": temp_result.get("fever_threshold_c", 37.5),
        "temp_video_details": temp_result,
        "annotated_image": temp_result.get("annotated_image"),
        "status_flags": status_flags,
        # Respiration fields (existing)
        "respiratory_rate": rr_result,
        "respiratory_rate_bpm": signal_rr,
        "respiratory_rate_used_bpm": computed_rr,
        "ml_respiratory_rate": ml_rr,
        "rr_signal": rr_signal,
        "total_breaths": rr_result.get("total_breaths"),
        "duration_seconds": rr_result.get("duration_seconds"),
        "roi_method": rr_result.get("roi_method"),
        "confidence": rr_result.get("confidence"),
        "rr_category": rr_result.get("rr_category"),
        "rr_error": rr_result.get("error"),
        # NEW: multi-method signal analysis results (display-only, does not affect ML or status flags)
        "signal_methods": signal_methods_result,
    }

def handle_analyze(req, debug):
    try:
        if 'image' not in req.files:
            return jsonify({"error": "No image part in the request"}), 400
            
        file = req.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        ext = _get_extension(file.filename)
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            if ext in ALLOWED_VIDEO_EXTENSIONS:
                return jsonify({"error": f"Invalid file type: received a video file (.{ext}). Use the /api/analyze/video endpoint for video files, or upload an image (.png, .jpg, .jpeg) to this endpoint."}), 400
            return jsonify({"error": f"Unsupported file type: .{ext}. Supported image formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"}), 400
            
        temp_filename = f"{uuid.uuid4().hex}.{ext}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        
        file.save(temp_path)
        
        result = analyze_single_image(temp_path, debug=debug)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if "error" in result:
            return jsonify(result), 500
            
        # Attach rule-based status flags
        temp_c = result.get("forehead_temp_c") or result.get("face_neck_temp_c")
        rr_bpm = req.form.get("respiratory_rate_bpm", type=float)
        
        result["status_flags"] = evaluate_vitals_status(temp_c, rr_bpm)
            
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5051, debug=True)
