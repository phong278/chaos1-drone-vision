from ultralytics import YOLO
import cv2
import os

# Load the ONNX model
model = YOLO("onnx/best.onnx", task="detect")  # ONNX is safer on Windows

# Input video path
input_video_path = "../vids/5.mp4"

# Output directory
output_dir = "../vids/detection/"
os.makedirs(output_dir, exist_ok=True)

# Create VideoCapture object
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'

# Output video path
output_path = os.path.join(output_dir, os.path.basename(input_video_path))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
print(f"Processing video: {input_video_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame, imgsz=640, conf=0.4, verbose=False)

    # Draw detections
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to output video
    out.write(frame)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print(f"Detection video saved to: {output_path}")
