#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
import socket
from gpiozero import Servo, Device
from gpiozero.pins.lgpio import LGPIOFactory
from collections import deque

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

# ================= PRECISION TRACKING =================
# Track object position history to detect movement
position_history = deque(maxlen=15)  # Last 15 positions for better smoothing
MOVEMENT_THRESHOLD = 10  # pixels - object must move this much to trigger tracking
CENTERING_THRESHOLD = 20  # pixels - how close to center before "locked"
LOCK_HOLD_TIME = 0.5  # seconds to wait before unlocking after centering

# Tracking states
STATE_SEARCHING = 0
STATE_CENTERING = 1
STATE_LOCKED = 2

tracking_state = STATE_SEARCHING
lock_time = 0
last_target_pos = None

# PD control variables
previous_error_x = 0
previous_error_y = 0
previous_time = time.time()

# Adaptive gains based on state
GAIN_CENTERING = 0.003  # Faster when centering (P gain)
GAIN_TRACKING = 0.002   # Normal tracking speed (P gain)
GAIN_LOCKED = 0.0008    # Much slower, smoother when locked (P gain)

# Derivative gain for damping (reduces overshoot)
GAIN_D = 0.005  # Dampens rapid movements (REDUCED - was causing oscillation)

# ================= SERVO TUNING =================
DEADZONE = 25  # Larger deadzone = less micro-movements
MAX_STEP = 0.03
MAX_STEP_LOCKED = 0.015  # Slower max step when locked

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

# ================= YOLO FUNCTION =================
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

# ================= MOVEMENT DETECTION =================
def is_object_moving():
    """Detect if object has moved significantly based on position history"""
    if len(position_history) < 3:
        return True  # Not enough data, assume moving
    
    # Calculate movement over last few frames
    recent_positions = list(position_history)[-5:]
    if len(recent_positions) < 2:
        return True
    
    max_delta = 0
    for i in range(len(recent_positions) - 1):
        dx = recent_positions[i+1][0] - recent_positions[i][0]
        dy = recent_positions[i+1][1] - recent_positions[i][1]
        delta = np.sqrt(dx*dx + dy*dy)
        max_delta = max(max_delta, delta)
    
    return max_delta > MOVEMENT_THRESHOLD

def is_centered(error_x, error_y):
    """Check if target is centered within threshold"""
    return abs(error_x) < CENTERING_THRESHOLD and abs(error_y) < CENTERING_THRESHOLD

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cx_frame, cy_frame = w // 2, h // 2
    now = time.time()

    # Run YOLO detection
    if now - last_yolo_time > YOLO_INTERVAL:
        last_yolo_time = now
        result = run_yolo(frame)
        if result:
            person_box = result
            last_person_time = now

    # Handle timeout
    if now - last_person_time > PERSON_TIMEOUT:
        person_box = None
        pan = tilt = roll = 0.0
        tracking_state = STATE_SEARCHING
        position_history.clear()

    # Process tracking
    should_move_motors = False
    
    if person_box:
        x1, y1, x2, y2, conf = person_box
        target_x = (x1 + x2) // 2
        target_y = (y1 + y2) // 2
        
        # Add to position history
        position_history.append((target_x, target_y))
        
        # Calculate error
        error_x = target_x - cx_frame
        error_y = target_y - cy_frame
        
        # State machine for precision tracking
        if tracking_state == STATE_SEARCHING:
            # Just found object, start centering
            tracking_state = STATE_CENTERING
            should_move_motors = True
            status_text = "CENTERING"
            status_color = (0, 165, 255)  # Orange
            
        elif tracking_state == STATE_CENTERING:
            # Check if centered
            if is_centered(error_x, error_y):
                tracking_state = STATE_LOCKED
                lock_time = now
                status_text = "LOCKED"
                status_color = (0, 255, 0)  # Green
            else:
                should_move_motors = True
                status_text = "CENTERING"
                status_color = (0, 165, 255)
                
        elif tracking_state == STATE_LOCKED:
            # Check if object is moving
            if is_object_moving():
                should_move_motors = True
                status_text = "LOCKED - TRACKING"
                status_color = (0, 255, 255)  # Yellow
            else:
                # Object stationary, check if still centered
                if not is_centered(error_x, error_y):
                    # Drifted off center, recenter
                    should_move_motors = True
                    status_text = "LOCKED - ADJUSTING"
                    status_color = (0, 255, 255)
                else:
                    should_move_motors = False
                    status_text = "LOCKED - STEADY"
                    status_color = (0, 255, 0)
        
        # Draw detection box and status
        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
        cv2.putText(frame, f"{status_text} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Draw center crosshair
        cv2.circle(frame, (target_x, target_y), 5, status_color, -1)
        
        # Draw centering zone
        cv2.rectangle(frame, 
                     (cx_frame - CENTERING_THRESHOLD, cy_frame - CENTERING_THRESHOLD),
                     (cx_frame + CENTERING_THRESHOLD, cy_frame + CENTERING_THRESHOLD),
                     (0, 255, 0), 1)
        
    else:
        target_x, target_y = cx_frame, cy_frame
        error_x = error_y = 0
        tracking_state = STATE_SEARCHING

    # Update servo positions based on state
    if should_move_motors:
        # Select gain and max step based on state
        if tracking_state == STATE_CENTERING:
            gain = GAIN_CENTERING
            max_step = MAX_STEP
        elif tracking_state == STATE_LOCKED:
            gain = GAIN_LOCKED
            max_step = MAX_STEP_LOCKED
        else:
            gain = GAIN_TRACKING
            max_step = MAX_STEP
        
        # Calculate time delta for derivative
        current_time = time.time()
        dt = current_time - previous_time
        if dt > 0 and dt < 0.5:  # Ignore large time gaps
            # Derivative term (rate of change of error)
            derivative_x = (error_x - previous_error_x) / dt
            derivative_y = (error_y - previous_error_y) / dt
            
            # Limit derivative to prevent spikes from detection jitter
            derivative_x = max(-100, min(100, derivative_x))
            derivative_y = max(-100, min(100, derivative_y))
        else:
            derivative_x = derivative_y = 0
        
        previous_error_x = error_x
        previous_error_y = error_y
        previous_time = current_time
        
        # PD control: P + D
        # P term: proportional to current error
        # D term: opposes rapid changes (damping)
        if abs(error_x) > DEADZONE:
            p_term = error_x * gain
            d_term = -derivative_x * GAIN_D  # Negative to dampen
            pan += max(-max_step, min(max_step, p_term + d_term))
        
        if abs(error_y) > DEADZONE:
            p_term = error_y * gain
            d_term = -derivative_y * GAIN_D
            tilt += max(-max_step, min(max_step, p_term + d_term))
    
    # Clamp servo values
    pan = max(-1.0, min(1.0, pan))
    tilt = max(-1.0, min(1.0, tilt))

    # Update servos
    pan_servo.value = pan
    tilt_servo.value = tilt
    roll_servo.value = 0.0

    # Draw center crosshair for frame
    cv2.line(frame, (cx_frame - 10, cy_frame), (cx_frame + 10, cy_frame), (255, 255, 255), 1)
    cv2.line(frame, (cx_frame, cy_frame - 10), (cx_frame, cy_frame + 10), (255, 255, 255), 1)

    with stream_lock:
        stream_frame = frame.copy()