"""
app.py - SafeDrive AI | Flask REST + MJPEG Streaming Server
Uses infer_single_image (MobileViT v2 single-frame classifier)
"""

import sys
import os
import time
import threading
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import requests


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from drowsiness_detection import (
    CONFIG,
    DrowsinessDetector,
    detect_and_crop_face,
    infer_single_image,
)

app = Flask(__name__)
CORS(app)

_lock = threading.Lock()
_latest = {
    "prediction": "Loading",
    "probability": 0.0,
    "alert": False,
    "latency_ms": 0.0,
    "fps": 0.0,
    "timestamp": "",
    "drowsy_seconds": 0.0,   # How long eyes have been continuously closed
}
_history = deque(maxlen=50)
_settings = {
    "alertThreshold": CONFIG["alert_threshold"],
    "inferInterval": 1,  # Run inference on every frame for max FPS
    "cameraResolution": "720p",
    "enableSound": True,
    "webhookUrl": "",
}
_frame_buffer = [None]       # JPEG display frames for MJPEG stream
_raw_frame_buffer = [None]   # Raw BGR frame shared with inference thread
_cam_debug = {"status": "initializing", "error": None, "frame_count": 0}

DEVICE = CONFIG["device"]

print("[API] Loading YOLO model on %s ..." % DEVICE)
model = DrowsinessDetector().to(DEVICE)
model.eval()



def trigger_webhook_async(url, payload):
    def run():
        try:
            print("[WEBHOOK] Sending alert POST to %s..." % url)
            res = requests.post(url, json=payload, timeout=3.0)
            print("[WEBHOOK] Trigger response code: %d" % res.status_code)
        except Exception as e:
            print("[WEBHOOK] Failed to trigger alert webhook: %s" % e)
    threading.Thread(target=run, daemon=True).start()


def camera_loop():
    global _cam_debug
    print("[CAM] Opening camera ...")
    _cam_debug["status"] = "opening_camera"

    cap = None
    # Try multiple camera indices and backends
    for cam_idx in [0, 1]:
        # On Windows, DSHOW is much more stable than MSMF/default for custom resolutions
        for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY), ("MSMF", cv2.CAP_MSMF)]:
            print("[CAM] Trying index=%d backend=%s ..." % (cam_idx, backend_name))
            try:
                cap = cv2.VideoCapture(cam_idx, backend)
                if cap.isOpened():
                    # Set resolution + FPS + minimal buffer for low latency
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test = cap.read()
                    if ret and test is not None:
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        print("[CAM] OK - opened index=%d backend=%s at 1280x720 @ %.0f FPS" % (cam_idx, backend_name, actual_fps))
                        break
                    
                    # If setting resolution failed to read, try default resolution
                    print("[CAM] BGR read failed at 1280x720, trying default resolution...")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test = cap.read()
                    if ret and test is not None:
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        print("[CAM] OK - opened index=%d backend=%s at 640x480 @ %.0f FPS" % (cam_idx, backend_name, actual_fps))
                        break

                    cap.release()
                    cap = None
                else:
                    cap.release()
                    cap = None
            except Exception as exc:
                print("[CAM] Exception trying index=%d backend=%s: %s" % (cam_idx, backend_name, exc))
                if cap is not None:
                    cap.release()
                    cap = None
        if cap is not None:
            break

    if cap is None:
        _cam_debug["status"] = "camera_failed"
        _cam_debug["error"] = "Could not open any camera"
        print("[CAM] FAILED: Could not open any camera (tried index 0 and 1)")
        return

    _cam_debug["status"] = "running"
    print("[CAM] Camera opened OK - starting capture loop")

    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0
    current_fps = 0

    while True:
        try:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            fps_counter += 1
            _cam_debug["frame_count"] = frame_count

            # Update FPS counter every second
            if time.time() - fps_timer > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
                # Push FPS into _latest so frontend sees it
                with _lock:
                    _latest["fps"] = current_fps

            # Share raw frame with inference thread (no copy needed — inference
            # thread always grabs the latest available frame)
            _raw_frame_buffer[0] = frame_bgr

            # Fetch latest inference results for HUD
            with _lock:
                pred_text  = _latest["prediction"]
                confidence = _latest["probability"]
                is_alert   = _latest["alert"]
                latency_ms = _latest["latency_ms"]

            # Draw HUD overlay
            display = frame_bgr.copy()
            h, w = display.shape[:2]

            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 88), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

            color = (0, 60, 255) if is_alert else (40, 220, 40)
            cv2.putText(display, "Status: %s" % pred_text, (20, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2, cv2.LINE_AA)

            info = "Conf: %.1f%%   Lat: %.0fms   FPS: %d" % (
                confidence * 100, latency_ms, current_fps)
            cv2.putText(display, info, (20, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            if is_alert:
                cv2.rectangle(display, (3, 3), (w - 3, h - 3), (0, 0, 255), 5)

            ok, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                _frame_buffer[0] = jpeg.tobytes()

        except Exception as e:
            _cam_debug["error"] = str(e)
            print("[CAM] Error: %s" % e)
            traceback.print_exc()
            time.sleep(0.1)

    cap.release()


def inference_loop():
    """Runs model inference in a separate thread so it never blocks the camera.
    
    Alert logic: Only triggers 'alert=True' after the model continuously
    predicts 'Drowsy' for DROWSY_ALERT_SECONDS seconds (default 5s).
    """
    DROWSY_ALERT_SECONDS = 5.0

    print("[INF] Inference thread started (alert after %.0fs of drowsiness)" % DROWSY_ALERT_SECONDS)
    last_alert_state = False
    drowsy_since = None   # timestamp when continuous drowsy prediction started

    while True:
        try:
            frame_bgr = _raw_frame_buffer[0]
            if frame_bgr is None:
                time.sleep(0.01)
                continue

            with _lock:
                threshold = float(_settings["alertThreshold"])

            face_rgb = detect_and_crop_face(frame_bgr)
            result = infer_single_image(model, face_rgb, DEVICE, threshold)

            pred_text  = result["prediction"]
            confidence = result["probability"]
            latency_ms = result["latency_ms"]
            now = time.time()

            # ── Sustained-drowsiness timer ──────────────────────────────
            if pred_text == "Drowsy":
                if drowsy_since is None:
                    drowsy_since = now          # start the clock
                drowsy_elapsed = now - drowsy_since
            else:
                drowsy_since = None             # reset: eyes opened
                drowsy_elapsed = 0.0

            # Alert only after DROWSY_ALERT_SECONDS of continuous drowsiness
            is_alert = drowsy_elapsed >= DROWSY_ALERT_SECONDS
            # ────────────────────────────────────────────────────────────

            # Trigger webhook on alert state transition False -> True
            if is_alert and not last_alert_state:
                with _lock:
                    webhook_url = _settings.get("webhookUrl", "")
                if webhook_url:
                    ts_alert = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    payload = {
                        "alert": True,
                        "prediction": pred_text,
                        "probability": confidence,
                        "timestamp": ts_alert,
                        "device": DEVICE,
                    }
                    trigger_webhook_async(webhook_url, payload)
            last_alert_state = is_alert

            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with _lock:
                _latest.update({
                    "prediction": pred_text,
                    "probability": confidence,
                    "alert": is_alert,
                    "latency_ms": latency_ms,
                    "timestamp": ts,
                    "drowsy_seconds": round(min(drowsy_elapsed, DROWSY_ALERT_SECONDS), 1),
                    "fps": _latest["fps"],
                })
                if is_alert:
                    _history.appendleft(dict(_latest))

        except Exception as e:
            print("[INF] Inference error: %s" % e)
            time.sleep(0.01)


threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()
print("[API] Camera + inference threads started")


_STREAM_INTERVAL = 1.0 / 30  # Exactly 30 FPS

def _gen_frames():
    while True:
        frame = _frame_buffer[0]
        if frame is None:
            time.sleep(0.005)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )
        time.sleep(_STREAM_INTERVAL)  # Exactly 30 FPS


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify(dict(_latest))

@app.route("/api/history")
def api_history():
    with _lock:
        return jsonify(list(_history))

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        with _lock:
            # Cast numeric types properly
            if "alertThreshold" in data:
                data["alertThreshold"] = float(data["alertThreshold"])
            if "inferInterval" in data:
                data["inferInterval"] = int(data["inferInterval"])
            _settings.update(data)
    with _lock:
        return jsonify({"success": True, "settings": dict(_settings)})

@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "model": CONFIG["cnn_backbone"],
        "model_loaded": MODEL_PATH.exists(),
    })

@app.route("/api/debug")
def api_debug():
    return jsonify({
        "camera": dict(_cam_debug),
        "frame_buffer_filled": _frame_buffer[0] is not None,
    })

if __name__ == "__main__":
    print("")
    print("=" * 60)
    print("  SafeDrive AI -- Flask API Server (MobileViT v2)")
    print("=" * 60)
    print("  Video stream : http://localhost:5000/video_feed")
    print("  Status JSON  : http://localhost:5000/api/status")
    print("  Debug info   : http://localhost:5000/api/debug")
    print("=" * 60)
    print("")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
