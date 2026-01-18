from ultralytics import YOLO
import cv2
import os

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "onnx/best.onnx"   # Path to your trained ONNX model
INPUT_IMAGE = "../media/3.jpg"  # Input image path
OUTPUT_DIR = "../media/detections/"
CONF_THRESHOLD = 0.4
IMG_SIZE = 640

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# LOAD MODEL
# ------------------------------
model = YOLO(MODEL_PATH, task="detect")  # Use task="detect" for object detection

# ------------------------------
# READ IMAGE
# ------------------------------
image = cv2.imread(INPUT_IMAGE)
if image is None:
    raise FileNotFoundError(f"Image not found: {INPUT_IMAGE}")

# ------------------------------
# RUN DETECTION
# ------------------------------
results = model(image, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)

# ------------------------------
# DRAW DETECTIONS
# ------------------------------
if results and results[0].boxes is not None:
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ------------------------------
# SAVE RESULT
# ------------------------------
output_path = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_IMAGE))
cv2.imwrite(output_path, image)
print(f"Detection saved to: {output_path}")
