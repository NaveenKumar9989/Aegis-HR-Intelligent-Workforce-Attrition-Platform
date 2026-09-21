"""
Aegis HR - Web Application Launcher.
Starts the backend REST API server and serves the frontend dashboard.
Usage:
    py -3.14 run_app.py
"""

import sys
import webbrowser
import threading
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import BEST_MODEL_PATH
from backend.app import app


def open_browser():
    """Wait for server to bind then launch the default web browser."""
    time.sleep(1.2)
    url = "http://127.0.0.1:5000"
    print(f"[+] Opening browser at: {url}")
    webbrowser.open(url)


def main():
    print("=" * 72)
    print("      AEGIS HR - INTELLIGENT WORKFORCE ATTRITION PLATFORM      ")
    print("=" * 72)

    if not BEST_MODEL_PATH.exists():
        print("[!] No trained model artifacts detected. Running training pipeline first...")
        from backend.pipeline.trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_and_evaluate()
        print("[+] Training completed!")

    print("[+] Server initializing at http://127.0.0.1:5000")
    print("[+] Press Ctrl+C in terminal to stop server.\n")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
