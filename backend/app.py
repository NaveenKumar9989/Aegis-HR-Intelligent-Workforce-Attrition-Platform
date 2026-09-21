"""
Flask REST API and Static File Server for Employee Attrition Platform.
Provides endpoints for health, dataset statistics, model metrics, retraining,
single prediction, batch CSV inference, and what-if simulation.
"""

import io
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd

from backend.config import BASE_DIR, METRICS_PATH, METADATA_PATH, DATA_PATH
from backend.pipeline.data_loader import DataLoader
from backend.pipeline.trainer import ModelTrainer
from backend.pipeline.predictor import AttritionPredictor

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path=""
)

# Initialize predictor singleton
predictor = AttritionPredictor()
data_loader = DataLoader()


@app.after_request
def add_cors_headers(response):
    """Enable CORS for modern browser interactions."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ==========================================
# Static Frontend Serving
# ==========================================
@app.route("/")
def serve_index():
    """Serve main frontend SPA application."""
    return send_from_directory(app.static_folder, "index.html")


# ==========================================
# API Endpoints
# ==========================================
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health status and model operational readiness."""
    is_ready = predictor.is_ready()
    return jsonify({
        "status": "healthy",
        "pipeline_ready": is_ready,
        "dataset_exists": DATA_PATH.exists()
    })


@app.route("/api/dataset/stats", methods=["GET"])
def get_dataset_stats():
    """Exploratory summary statistics for executive charts."""
    try:
        stats = data_loader.get_dataset_stats()
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/metrics", methods=["GET"])
def get_model_metrics():
    """Returns benchmark comparison metrics across all trained models."""
    try:
        if not METRICS_PATH.exists():
            return jsonify({
                "success": False,
                "error": "Models have not been trained yet. Please train models first."
            }), 404

        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        return jsonify({"success": True, "data": metrics_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/train", methods=["POST"])
def trigger_training():
    """Triggers end-to-end retraining of models."""
    try:
        trainer = ModelTrainer()
        report = trainer.train_and_evaluate()
        # Reload predictor with fresh artifacts
        global predictor
        predictor = AttritionPredictor()
        return jsonify({
            "success": True,
            "message": "Model training completed successfully.",
            "data": report
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_single():
    """Single employee risk scoring and prescriptive recommendations."""
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"success": False, "error": "Empty JSON payload received."}), 400

        result = predictor.predict_single(payload)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """Batch employee inference via CSV file upload or JSON list."""
    try:
        if "file" in request.files:
            uploaded_file = request.files["file"]
            if uploaded_file.filename == "":
                return jsonify({"success": False, "error": "No file selected."}), 400
            df = pd.read_csv(uploaded_file)
        elif request.is_json:
            payload = request.get_json(force=True)
            if not isinstance(payload, list):
                return jsonify({"success": False, "error": "Expected a JSON list of records."}), 400
            df = pd.DataFrame(payload)
        else:
            return jsonify({"success": False, "error": "Expected either a CSV file or JSON array."}), 400

        predictions = predictor.predict_batch(df)

        # Compute batch summary metrics
        total = len(predictions)
        high_risk_count = sum(1 for p in predictions if p["risk_level"] in ["HIGH", "CRITICAL"])
        attrited_pred_count = sum(1 for p in predictions if p["attrition_prediction"] == 1)
        avg_prob = sum(p["probability"] for p in predictions) / total if total > 0 else 0.0

        return jsonify({
            "success": True,
            "summary": {
                "total_records": total,
                "high_or_critical_risk_count": high_risk_count,
                "predicted_attrition_count": attrited_pred_count,
                "average_risk_percentage": round(avg_prob * 100, 1)
            },
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/simulate", methods=["POST"])
def simulate_what_if():
    """What-If simulation comparing baseline vs policy adjustments."""
    try:
        body = request.get_json(force=True)
        base_data = body.get("base")
        modifications = body.get("modifications")

        if not base_data or not modifications:
            return jsonify({
                "success": False,
                "error": "Both 'base' and 'modifications' objects are required in JSON body."
            }), 400

        sim_result = predictor.simulate_what_if(base_data, modifications)
        return jsonify({"success": True, "data": sim_result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


def create_app():
    return app


if __name__ == "__main__":
    print("[+] Aegis HR API server launching on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
