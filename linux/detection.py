#!/usr/bin/env python3
import cv2
import time
import math
from ultralytics import YOLO

# ---------------- GPIO SAFE IMPORT ----------------
try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None
    print("⚠️ GPIO not available")

# -------------------------------------------------
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except:
        return None

# ================== DETECTION SYSTEM ==================
class DetectionSystem:
    def __init__(self, config):
        self.config = config

        # -------- MODEL --------
        self.model = YOLO("models/openvino/")
        self.model.conf = config["detection"]["confidence"]

        # -------- CAMERA --------
        self.cap = cv2.VideoCapture(config["camera"]["device_id"], cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
        self.cap.set(cv2.CAP_PROP_FPS, config["camera"]["fps"])

        # -------- GIMBAL --------
        self.pan_pin = 17
        self.tilt_pin = 27
        self.pan_angle = 90
        self.tilt_angle = 90
        self.max_angle = 180
        self.min_angle = 0
        self.kp = 0.1

        if GPIO:
            self.setup_motors()

    # ---------------- GPIO ----------------
    def setup_motors(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pan_pin, GPIO.OUT)
        GPIO.setup(self.tilt_pin, GPIO.OUT)
        self.pan_pwm = GPIO.PWM(self.pan_pin, 50)
        self.tilt_pwm = GPIO.PWM(self.tilt_pin, 50)
        self.pan_pwm.start(7.5)
        self.tilt_pwm.start(7.5)

    def move_motors(self, x_offset, y_offset):
        if not GPIO:
            return

        self.pan_angle += self.kp * x_offset
        self.tilt_angle -= self.kp * y_offset

        self.pan_angle = max(self.min_angle, min(self.max_angle, self.pan_angle))
        self.tilt_angle = max(self.min_angle, min(self.max_angle, self.tilt_angle))

        def angle_to_duty(a):
            return 2.5 + (a / 180.0) * 10.0

        self.pan_pwm.ChangeDutyCycle(angle_to_duty(self.pan_angle))
        self.tilt_pwm.ChangeDutyCycle(angle_to_duty(self.tilt_angle))

    # ---------------- TRACKING ----------------
    def track_object(self, detections, frame):
        if not detections:
            return

        det = detections[0]
        x1, y1, x2, y2 = det["bbox"]
        obj_x = (x1 + x2) / 2
        obj_y = (y1 + y2) / 2

        frame_cx = frame.shape[1] / 2
        frame_cy = frame.shape[0] / 2

        self.move_motors(obj_x - frame_cx, obj_y - frame_cy)

    # ---------------- DETECTION ----------------
    def detect(self, frame):
        results = self.model(
            frame,
            imgsz=self.config["performance"]["inference_size"],
            conf=self.config["detection"]["confidence"],
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < self.config["detection"]["confidence"]:
                    continue
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                detections.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "bbox": box.xyxy[0].cpu().numpy().astype(int),
                    }
                )
        return detections

    # ---------------- DRAW ----------------
    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']} {det['confidence']:.1%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return frame

    # ---------------- MAIN LOOP ----------------
    def run(self):
        print("Starting Object Detection (local preview)")
        frame_count = 0
        fps_time = time.time()
        fps = 0
        last_detections = []

        try:
            while True:
                start = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera error")
                    break

                frame_count += 1

                if frame_count % (self.config["performance"]["frame_skip"] + 1) == 0:
                    last_detections = self.detect(frame)

                self.track_object(last_detections, frame)

                display_frame = frame
                self.draw_detections(display_frame, last_detections)

                if time.time() - fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    fps_time = time.time()

                cv2.putText(
                    display_frame,
                    f"FPS: {fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # ---- LOCAL PREVIEW (FAST) ----
                cv2.imshow("YOLO Preview", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                elapsed = time.time() - start
                target = 1.0 / self.config["performance"]["throttle_fps"]
                if elapsed < target:
                    time.sleep(target - elapsed)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if GPIO:
                GPIO.cleanup()

# ================== MAIN ==================
if __name__ == "__main__":
    config = {
        "performance": {
            "throttle_fps": 10,
            "frame_skip": 4,
            "inference_size": 640,  # MUST stay 640
        },
        "camera": {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 10,
        },
        "detection": {
            "confidence": 0.4,
        },
    }

    DetectionSystem(config).run()
