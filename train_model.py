"""
Road Pavement Detection System — Python ML Pipeline
Uses YOLOv8 for defect detection + severity scoring (PCI index)
Install: pip install ultralytics opencv-python numpy flask torch torchvision
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from flask import Flask, request, jsonify
import base64
import json

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DEFECT_CLASSES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
    "rutting",
    "raveling",
    "edge_break"
]

# Severity weights per defect type (used for PCI scoring)
SEVERITY_WEIGHTS = {
    "longitudinal_crack": 0.4,
    "transverse_crack":   0.4,
    "alligator_crack":    0.7,
    "pothole":            1.0,
    "rutting":            0.8,
    "raveling":           0.5,
    "edge_break":         0.6,
}

MODEL_PATH = "road_defect_yolov8.pt"
CONFIDENCE_THRESHOLD = 0.45
EXPORT_ONNX = True  # Export for C++ / browser use


# ─────────────────────────────────────────────
#  DATASET PREPARATION
# ─────────────────────────────────────────────
def create_dataset_yaml(dataset_dir: str) -> str:
    """
    Creates YAML config for YOLOv8 training.
    dataset_dir should contain: images/train, images/val, labels/train, labels/val
    """
    yaml_content = f"""
path: {dataset_dir}
train: images/train
val:   images/val

nc: {len(DEFECT_CLASSES)}
names: {DEFECT_CLASSES}
"""
    yaml_path = os.path.join(dataset_dir, "road_defects.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[+] Dataset YAML written to: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
def train_model(dataset_yaml: str, epochs: int = 50, img_size: int = 640):
    """
    Fine-tune YOLOv8n on road defect dataset.
    Uses transfer learning from COCO pretrained weights.
    """
    print("\n" + "="*50)
    print("  ROAD DEFECT DETECTOR — Training")
    print("="*50)

    # Load pretrained YOLOv8n (nano — fast inference, good for edge)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        augment=True,          # random flip, mosaic, HSV
        degrees=5.0,           # slight rotation augment
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        name="road_defect_v1",
        project="runs/detect",
        save=True,
        plots=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Save best weights
    best_weights = Path("runs/detect/road_defect_v1/weights/best.pt")
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, MODEL_PATH)
        print(f"\n[+] Best model saved to: {MODEL_PATH}")

    # Export to ONNX for C++ / browser inference
    if EXPORT_ONNX:
        model.export(format="onnx", imgsz=img_size, simplify=True)
        print("[+] ONNX model exported for C++ and browser use")

    return results


# ─────────────────────────────────────────────
#  PCI SEVERITY SCORING
# ─────────────────────────────────────────────
def compute_pci(detections: list, image_area: int) -> dict:
    """
    Compute Pavement Condition Index (0–100, higher = better condition).
    Formula based on ASTM D6433 simplified for ML output.
    """
    if not detections:
        return {"pci": 100, "rating": "Excellent", "color": "#22c55e"}

    total_deduct = 0.0
    defect_counts = {}

    for det in detections:
        cls = det["class"]
        conf = det["confidence"]
        bbox = det["bbox"]  # [x1, y1, x2, y2]

        # Area of defect as % of image
        defect_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_ratio  = defect_area / max(image_area, 1)

        weight = SEVERITY_WEIGHTS.get(cls, 0.5)
        deduct = weight * conf * area_ratio * 100

        total_deduct += deduct
        defect_counts[cls] = defect_counts.get(cls, 0) + 1

    # Clamp PCI between 0–100
    pci = max(0, min(100, 100 - total_deduct * 10))

    # Rating bands
    if pci >= 85:
        rating, color = "Excellent", "#22c55e"
    elif pci >= 70:
        rating, color = "Good",      "#84cc16"
    elif pci >= 55:
        rating, color = "Fair",      "#eab308"
    elif pci >= 40:
        rating, color = "Poor",      "#f97316"
    elif pci >= 25:
        rating, color = "Very Poor", "#ef4444"
    else:
        rating, color = "Failed",    "#991b1b"

    return {
        "pci":           round(pci, 1),
        "rating":        rating,
        "color":         color,
        "defect_counts": defect_counts,
        "total_defects": len(detections)
    }


# ─────────────────────────────────────────────
#  INFERENCE ENGINE
# ─────────────────────────────────────────────
class RoadDefectDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        print(f"[+] Loading model from: {model_path}")
        # Fall back to pretrained COCO model for demo if custom weights missing
        if not os.path.exists(model_path):
            print("[!] Custom weights not found — using YOLOv8n pretrained (demo mode)")
            model_path = "yolov8n.pt"
        self.model = YOLO(model_path)
        self.model.conf = CONFIDENCE_THRESHOLD
        print("[+] Model ready")

    def predict(self, image: np.ndarray) -> dict:
        """Run inference on a single BGR image (OpenCV format)."""
        h, w = image.shape[:2]
        results = self.model(image, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = (DEFECT_CLASSES[cls_id]
                        if cls_id < len(DEFECT_CLASSES)
                        else results.names[cls_id])
            det = {
                "class":      cls_name,
                "confidence": round(float(box.conf[0]), 3),
                "bbox":       [round(x) for x in box.xyxy[0].tolist()]
            }
            detections.append(det)

        severity = compute_pci(detections, w * h)

        # Draw annotated frame
        annotated = results.plot(
            line_width=2,
            font_size=0.5,
            labels=True,
            conf=True
        )

        return {
            "detections": detections,
            "severity":   severity,
            "frame":      annotated   # BGR numpy array
        }

    def process_video(self, video_path: str, output_path: str = "output.mp4"):
        """Process a full video file and save annotated output."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_n    = 0
        all_scores = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference every 3rd frame for speed
            if frame_n % 3 == 0:
                result = self.predict(frame)
                all_scores.append(result["severity"]["pci"])
                annotated = result["frame"]
            out.write(annotated)
            frame_n += 1

        cap.release()
        out.release()

        avg_pci = round(np.mean(all_scores), 1) if all_scores else 100
        print(f"\n[+] Video processed: {frame_n} frames")
        print(f"[+] Average PCI: {avg_pci}")
        print(f"[+] Output: {output_path}")
        return avg_pci


# ─────────────────────────────────────────────
#  FLASK REST API  (consumed by HTML dashboard)
# ─────────────────────────────────────────────
app      = Flask(__name__)
detector = None   # lazy-loaded on first request

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON: { "image": "<base64 JPEG>" }
    Returns: detections, severity scores, annotated image
    """
    global detector
    if detector is None:
        detector = RoadDefectDetector()

    data   = request.get_json()
    b64    = data.get("image", "")
    img_bytes = base64.b64decode(b64)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    result      = detector.predict(frame)
    _, enc_buf  = cv2.imencode(".jpg", result["frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(enc_buf.tobytes()).decode()

    return jsonify({
        "detections":      result["detections"],
        "severity":        result["severity"],
        "annotated_image": annotated_b64
    })

@app.route("/predict_url", methods=["POST"])
def predict_url():
    """Accept image URL instead of base64."""
    import urllib.request
    global detector
    if detector is None:
        detector = RoadDefectDetector()

    url       = request.get_json().get("url", "")
    req       = urllib.request.urlopen(url)
    nparr     = np.frombuffer(req.read(), np.uint8)
    frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result    = detector.predict(frame)
    _, enc    = cv2.imencode(".jpg", result["frame"])
    return jsonify({
        "detections":      result["detections"],
        "severity":        result["severity"],
        "annotated_image": base64.b64encode(enc.tobytes()).decode()
    })


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # python train_model.py train /path/to/dataset
        dataset_dir  = sys.argv[2] if len(sys.argv) > 2 else "./dataset"
        yaml_path    = create_dataset_yaml(dataset_dir)
        train_model(yaml_path, epochs=50)

    elif len(sys.argv) > 1 and sys.argv[1] == "video":
        # python train_model.py video road.mp4
        det = RoadDefectDetector()
        det.process_video(sys.argv[2])

    else:
        # python train_model.py  →  start API server
        print("\n Road Pavement Detection API")
        print(" POST /predict  { image: <base64> }")
        print(" GET  /health\n")
        app.run(host="0.0.0.0", port=5000, debug=False)
