import cv2
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "models/onnx/best.onnx"  # or your OpenVINO/ONNX model
CONFIDENCE = 0.4
CAMERA_ID = 0  # default laptop webcam

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)
model.conf = CONFIDENCE

# ---------------- OPEN CAMERA ----------------
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("🟢 Camera opened. Press 'q' to quit.")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Run detection
    results = model(frame)

    # Draw detections
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id] if class_id < len(model.names) else "object"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.1%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
