#!/usr/bin/env python3
import cv2
import time
from ultralytics import YOLO
from gpiozero import Servo, Device
from gpiozero.pins.lgpio import LGPIOFactory

# ================= GPIO =================
Device.pin_factory = LGPIOFactory()
pan_servo  = Servo(17, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
tilt_servo = Servo(27, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

pan = 0.0
tilt = 0.0

DEADZONE = 15
GAIN = 0.002
MAX_STEP = 0.03

# ================= CUSTOM CLASS NAMES =================
CLASS_NAMES = [
    "artillery", "missile", "radar", "m. rocket launcher", "soldier",
    "tank", "vehicle", "aircraft", "ships"
]

# ================== DETECTION SYSTEM ==================
class DetectionSystem:
    def __init__(self, config):
        self.config = config

        # ---------- LOAD ONNX MODEL ----------
        self.model = YOLO("models/onnx/best.onnx", task="detect")
        self.model.conf = config["detection"]["confidence"]

        # ---------- CAMERA ----------
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.last_detections = []

    # ---------------- DETECTION ----------------
    def detect(self, frame):
        results = self.model(
            frame,
            imgsz=640,
            conf=self.config["detection"]["confidence"],
            verbose=False
        )

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < self.config["detection"]["confidence"]:
                    continue
                class_id = int(box.cls[0])
                # Only track the first class in CLASS_NAMES
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append((x1, y1, x2, y2, conf))

        return detections

    # ---------------- TRACKING ----------------
    def track_object(self, detections, frame):
        global pan, tilt

        if not detections:
            return

        x1, y1, x2, y2, _ = detections[0]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        h, w, _ = frame.shape
        cx_frame = w // 2
        cy_frame = h // 2

        error_x = cx - cx_frame
        error_y = cy - cy_frame

        if abs(error_x) > DEADZONE:
            pan += max(-MAX_STEP, min(MAX_STEP, error_x * GAIN))

        if abs(error_y) > DEADZONE:
            tilt += max(-MAX_STEP, min(MAX_STEP, error_y * GAIN))

        pan = max(-1.0, min(1.0, pan))
        tilt = max(-1.0, min(1.0, tilt))

        pan_servo.value = pan
        tilt_servo.value = tilt

    # ---------------- MAIN LOOP ----------------
    def run(self):
        frame_count = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                frame_count += 1

                # Run detection on every Nth frame
                if frame_count % (self.config["performance"]["frame_skip"] + 1) == 0:
                    self.last_detections = self.detect(frame)

                self.track_object(self.last_detections, frame)

                # Draw bounding boxes
                for x1, y1, x2, y2, conf in self.last_detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{CLASS_NAMES[0]} {conf:.2f}"  # Only first class
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw center marker
                h, w, _ = frame.shape
                cv2.drawMarker(frame, (w//2, h//2), (255,255,255), cv2.MARKER_CROSS, 20, 2)

                # Show frame
                cv2.imshow("YOLO Servo Tracker", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

# ================== MAIN ==================
if __name__ == "__main__":
    config = {
        "performance": {
            "frame_skip": 4,
        },
        "detection": {
            "confidence": 0.2,
        },
    }

    DetectionSystem(config).run()
