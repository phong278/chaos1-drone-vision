#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
import socket
from gpiozero import Servo, Device
from gpiozero.pins.lgpio import LGPIOFactory

# ================= GPIO =================
Device.pin_factory = LGPIOFactory()

pan_servo  = Servo(18, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
tilt_servo = Servo(13, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
roll_servo = Servo(12, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

pan = tilt = roll = 0.0

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ================= YOLOv8 ONNX =================
MODEL_PATH = "yolov8n.onnx"
YOLO_CONF = 0.4
YOLO_INTERVAL = 0.1
PERSON_TIMEOUT = 1.0

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
yolo_input = sess.get_inputs()[0].name

last_yolo_time = 0
last_person_time = 0
person_box = None

# ================= SERVO TUNING =================
DEADZONE = 15
GAIN = 0.002
MAX_STEP = 0.03

# ================= STREAMING =================
STREAM_PORT = 5000
stream_frame = None
stream_lock = threading.Lock()

def mjpeg_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", STREAM_PORT))
    server.listen(1)
    print(f"🌐 Stream: http://<pi-ip>:{STREAM_PORT}")

    while True:
        conn, _ = server.accept()
        try:
            conn.sendall(
                b"HTTP/1.0 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n"
            )

            while True:
                with stream_lock:
                    frame = None if stream_frame is None else stream_frame.copy()

                if frame is None:
                    time.sleep(0.05)
                    continue

                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if not ok:
                    continue

                conn.sendall(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                conn.sendall(jpg.tobytes())
                conn.sendall(b"\r\n")
                time.sleep(0.1)
        except:
            pass
        finally:
            conn.close()

threading.Thread(target=mjpeg_server, daemon=True).start()

# ================= YOLO FUNCTION (WORKING ONE) =================
def run_yolo(frame):
    h, w, _ = frame.shape

    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    output = sess.run(None, {yolo_input: img})[0]
    output = np.squeeze(output)

    boxes = output[:4, :]
    scores = output[4:, :]

    class_ids = np.argmax(scores, axis=0)
    confidences = scores[class_ids, np.arange(scores.shape[1])]

    best = None
    best_conf = 0.0

    for i in range(scores.shape[1]):
        if class_ids[i] == 0 and confidences[i] > YOLO_CONF:
            if confidences[i] > best_conf:
                cx, cy, bw, bh = boxes[:, i]
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                best = (x1, y1, x2, y2, confidences[i])
                best_conf = confidences[i]

    return best

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cx_frame, cy_frame = w // 2, h // 2
    now = time.time()

    if now - last_yolo_time > YOLO_INTERVAL:
        last_yolo_time = now
        result = run_yolo(frame)
        if result:
            person_box = result
            last_person_time = now

    if now - last_person_time > PERSON_TIMEOUT:
        person_box = None
        pan = tilt = roll = 0.0

    if person_box:
        x1, y1, x2, y2, conf = person_box
        target_x = (x1 + x2) // 2
        target_y = (y1 + y2) // 2

        # ✅ DRAW FOR STREAM
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"PERSON {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        target_x, target_y = cx_frame, cy_frame

    error_x = target_x - cx_frame
    error_y = target_y - cy_frame

    if abs(error_x) > DEADZONE:
        pan += max(-MAX_STEP, min(MAX_STEP, error_x * GAIN))
    if abs(error_y) > DEADZONE:
        tilt += max(-MAX_STEP, min(MAX_STEP, error_y * GAIN))

    pan = max(-1.0, min(1.0, pan))
    tilt = max(-1.0, min(1.0, tilt))

    pan_servo.value = pan
    tilt_servo.value = tilt
    roll_servo.value = 0.0

    with stream_lock:
        stream_frame = frame.copy()
