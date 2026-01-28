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

# ROLL AXIS: INACTIVE - Reserved for future FC integration
# Uncomment below if you need roll stabilization from flight controller
# roll_servo = Servo(12, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
# roll_servo.value = 0.0

pan = tilt = 0.0

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ================= YOLOv8 ONNX =================
MODEL_PATH = "yolov8n.onnx"
YOLO_CONF = 0.4
YOLO_INTERVAL = 0.10  # 10 Hz detection
PERSON_TIMEOUT = 1.0

# YOLO Export Command (for reference):
# yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True
# Expected output shape: [1, 84, 8400] where 84 = [cx, cy, w, h, 80 classes]

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
yolo_input = sess.get_inputs()[0].name
yolo_output_shape = sess.get_outputs()[0].shape  # Store for validation

last_yolo_time = 0
last_person_time = 0
person_box = None

# ================= VELOCITY PREDICTION =================
# Track smoothed position and velocity (not raw YOLO)
smoothed_position = None
smoothed_velocity = (0, 0)
position_smoothing = 0.5   # HIGHER = more responsive, less lag (was 0.15)
velocity_smoothing = 0.6   # HIGHER = faster velocity tracking (was 0.3)
predicted_latency = 0.15   # System latency estimate (150ms)

def estimate_velocity():
    """Return current smoothed velocity"""
    return smoothed_velocity

def update_position_velocity(detected_x, detected_y, dt):
    """Update smoothed position and velocity using EMA"""
    global smoothed_position, smoothed_velocity
    
    if smoothed_position is None:
        smoothed_position = (detected_x, detected_y)
        smoothed_velocity = (0, 0)
        return
    
    # Calculate raw velocity
    if dt > 0:
        vx_raw = (detected_x - smoothed_position[0]) / dt
        vy_raw = (detected_y - smoothed_position[1]) / dt
    else:
        vx_raw, vy_raw = 0, 0
    
    # Smooth position (EMA filter)
    new_x = position_smoothing * detected_x + (1 - position_smoothing) * smoothed_position[0]
    new_y = position_smoothing * detected_y + (1 - position_smoothing) * smoothed_position[1]
    smoothed_position = (new_x, new_y)
    
    # Smooth velocity (EMA filter)
    vx_smooth = velocity_smoothing * vx_raw + (1 - velocity_smoothing) * smoothed_velocity[0]
    vy_smooth = velocity_smoothing * vy_raw + (1 - velocity_smoothing) * smoothed_velocity[1]
    smoothed_velocity = (vx_smooth, vy_smooth)

def predict_position(current_x, current_y, vx, vy):
    """Predict where target will be after latency"""
    predicted_x = current_x + vx * predicted_latency
    predicted_y = current_y + vy * predicted_latency
    return int(predicted_x), int(predicted_y)

# ================= TRACKING STATE =================
position_history = deque(maxlen=5)  # For velocity calculation
last_position = None
last_position_time = None

STATE_SEARCHING = 0
STATE_CENTERING = 1
STATE_LOCKED = 2

tracking_state = STATE_SEARCHING
lock_time = 0

# ================= PD CONTROL WITH DEADZONE =================
previous_error_x = 0
previous_error_y = 0
previous_time = time.time()

# Separated gains for different states
GAIN_CENTERING = 0.005   # Fast initial lock
GAIN_TRACKING = 0.004    # Moderate tracking
GAIN_LOCKED = 0.002      # Gentle corrections when locked

GAIN_D = 0.035  # Moderate damping

# CRITICAL: Deadzone to avoid chasing noise
DEADZONE = 3  # REDUCED - allows continuous fine adjustments (was 8)

# Thresholds
CENTERING_THRESHOLD = 18
MAX_STEP = 0.035
MAX_STEP_LOCKED = 0.025  # Slower when locked

# FPS tracking
frame_count = 0
fps_start_time = time.time()
current_fps = 0

# Debug printing
last_print_time = 0
PRINT_INTERVAL = 0.5  # Print debug info every 0.5 seconds

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
    
    # VALIDATION: Check output shape
    # Expected: [84, 8400] for YOLOv8 (4 box coords + 80 classes)
    if output.shape[0] not in [84, 85]:  # Some exports have 85
        print(f"⚠️ WARNING: Unexpected YOLO output shape: {output.shape}")
        print(f"Expected [84 or 85, N]. Check export command.")

    boxes = output[:4, :]  # [cx, cy, w, h]
    scores = output[4:, :]  # [80 class scores]

    class_ids = np.argmax(scores, axis=0)
    confidences = scores[class_ids, np.arange(scores.shape[1])]

    # Collect ALL person detections (class 0 = person in COCO)
    detections = []
    for i in range(scores.shape[1]):
        if class_ids[i] == 0 and confidences[i] > YOLO_CONF:
            cx, cy, bw, bh = boxes[:, i]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detections.append({
                'box': (x1, y1, x2, y2),
                'conf': confidences[i],
                'center': (center_x, center_y)
            })
    
    return detections

def select_best_person(detections, current_target):
    """Select which person to track"""
    if not detections:
        return None
    
    if current_target is not None:
        cx_current, cy_current = current_target
        min_distance = float('inf')
        best_detection = None
        
        for det in detections:
            cx, cy = det['center']
            distance = np.sqrt((cx - cx_current)**2 + (cy - cy_current)**2)
            if distance < min_distance:
                min_distance = distance
                best_detection = det
        
        if min_distance < 100:
            return best_detection
    
    # Pick largest person (closest to camera)
    best_detection = max(detections, key=lambda d: 
        (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]))
    
    return best_detection

def is_centered(error_x, error_y):
    """Check if target is centered"""
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

    # Run YOLO detection periodically
    all_detections = []  # Store all detections for visualization
    if now - last_yolo_time > YOLO_INTERVAL:
        last_yolo_time = now
        detections = run_yolo(frame)
        all_detections = detections  # Keep for drawing
        
        current_target_pos = None
        if person_box:
            x1, y1, x2, y2, conf = person_box
            current_target_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        best_person = select_best_person(detections, current_target_pos)
        
        if best_person:
            x1, y1, x2, y2 = best_person['box']
            person_box = (x1, y1, x2, y2, best_person['conf'])
            last_person_time = now

    # Handle timeout - Gradual filter decay instead of hard reset
    if now - last_person_time > PERSON_TIMEOUT:
        person_box = None
        tracking_state = STATE_SEARCHING
        position_history.clear()
        
        # Decay velocity gradually instead of hard reset
        # This prevents "lunge" when target reappears
        if smoothed_velocity != (0, 0):
            vx, vy = smoothed_velocity
            smoothed_velocity = (vx * 0.8, vy * 0.8)  # Decay by 20% per timeout
            
        # Keep smoothed_position for continuity
        # Only reset if timeout is very long (3+ seconds)
        if last_person_time > 0 and now - last_person_time > 3.0:
            smoothed_position = None
            smoothed_velocity = (0, 0)
        
        last_position = None
        # NOTE: pan/tilt NOT reset - holds last position

    # Process tracking with PREDICTION
    should_move_motors = False
    tracked_center = None  # Store center of tracked person
    current_gain = 0  # Initialize here so it's always defined
    current_max_step = 0
    
    if person_box:
        x1, y1, x2, y2, conf = person_box
        detected_x = (x1 + x2) // 2
        detected_y = (y1 + y2) // 2
        tracked_center = (detected_x, detected_y)  # For comparison later
        
        # Update smoothed position and velocity
        if last_position_time is not None:
            dt = now - last_position_time
            update_position_velocity(detected_x, detected_y, dt)
        else:
            smoothed_position = (detected_x, detected_y)
            smoothed_velocity = (0, 0)
        
        last_position_time = now
        
        # PREDICT using smoothed position and velocity
        vx, vy = estimate_velocity()
        if smoothed_position:
            smooth_x, smooth_y = smoothed_position
            target_x, target_y = predict_position(smooth_x, smooth_y, vx, vy)
        else:
            target_x, target_y = detected_x, detected_y
        
        # Clamp predicted position to frame
        target_x = max(0, min(w, target_x))
        target_y = max(0, min(h, target_y))
        
        # Calculate error from predicted position
        error_x = target_x - cx_frame
        error_y = target_y - cy_frame
        
        # State machine
        if tracking_state == STATE_SEARCHING:
            tracking_state = STATE_CENTERING
            should_move_motors = True
            status_text = "CENTERING"
            status_color = (0, 165, 255)
            
        elif tracking_state == STATE_CENTERING:
            should_move_motors = True
            if is_centered(error_x, error_y):
                tracking_state = STATE_LOCKED
                lock_time = now
                status_text = "LOCKED"
                status_color = (0, 255, 0)
            else:
                status_text = "CENTERING"
                status_color = (0, 165, 255)
                
        elif tracking_state == STATE_LOCKED:
            # ALWAYS track when locked - no more "steady" state
            # This ensures continuous centering even with jitter
            should_move_motors = True
            status_text = "LOCKED - TRACKING"
            status_color = (0, 255, 0)  # Green when locked
        
        # === SELECT GAIN BEFORE DEBUG PRINT ===
        # Do this ONCE per frame, right after state machine
        if tracking_state == STATE_CENTERING:
            current_gain = GAIN_CENTERING
            current_max_step = MAX_STEP
        elif tracking_state == STATE_LOCKED:
            current_gain = GAIN_LOCKED
            current_max_step = MAX_STEP_LOCKED
        else:
            current_gain = GAIN_TRACKING
            current_max_step = MAX_STEP
        
        # Distance-based gain scaling (expert improvement)
        error_mag = np.sqrt(error_x**2 + error_y**2)
        gain_scale = min(1.0, error_mag / 80.0)
        effective_gain = current_gain * gain_scale
        
        # === DRAW ALL DETECTED PEOPLE ===
        # First, draw all detections in gray (non-tracked people)
        for det in all_detections:
            det_center = det['center']
            # Check if this is the tracked person
            is_tracked = (tracked_center and 
                         abs(det_center[0] - tracked_center[0]) < 10 and
                         abs(det_center[1] - tracked_center[1]) < 10)
            
            if not is_tracked:
                # Draw non-tracked people in gray
                dx1, dy1, dx2, dy2 = det['box']
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (128, 128, 128), 2)
                cv2.putText(frame, f"PERSON {det['conf']:.2f}",
                           (dx1, dy1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Then draw tracked person in color (on top)
        cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 3)  # Thicker border
        cv2.putText(frame, f"TRACKING: {status_text} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Draw detected center (blue)
        cv2.circle(frame, (detected_x, detected_y), 5, (255, 0, 0), -1)
        cv2.putText(frame, "RAW", (detected_x + 8, detected_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Draw smoothed position (cyan)
        if smoothed_position:
            smooth_x, smooth_y = int(smoothed_position[0]), int(smoothed_position[1])
            cv2.circle(frame, (smooth_x, smooth_y), 5, (255, 255, 0), -1)
            cv2.putText(frame, "SMOOTH", (smooth_x + 8, smooth_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Draw predicted center (green) - where we're actually tracking
        if smoothed_position and (abs(vx) > 1 or abs(vy) > 1):
            cv2.circle(frame, (target_x, target_y), 5, (0, 255, 0), -1)
            cv2.line(frame, (smooth_x, smooth_y), (target_x, target_y), (0, 255, 0), 2)
            cv2.putText(frame, "PREDICT", (target_x + 8, target_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Draw error visualization - line from target to frame center
        cv2.line(frame, (target_x, target_y), (cx_frame, cy_frame), (255, 0, 255), 2)
        cv2.putText(frame, f"Error: {int(abs(error_x))}px, {int(abs(error_y))}px", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # === DEBUG PRINTING ===
        if now - last_print_time > PRINT_INTERVAL:
            last_print_time = now
            
            # Calculate total error magnitude
            total_error = np.sqrt(error_x**2 + error_y**2)
            
            print(f"\n{'='*60}")
            print(f"TRACKING DIAGNOSTICS @ {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            print(f"State: {['SEARCHING', 'CENTERING', 'LOCKED'][tracking_state]}")
            print(f"FPS: {current_fps:.1f} Hz")
            print(f"\n--- POSITIONS ---")
            print(f"Frame Center:    ({cx_frame}, {cy_frame})")
            print(f"Raw Detection:   ({detected_x}, {detected_y})")
            if smoothed_position:
                print(f"Smoothed:        ({int(smooth_x)}, {int(smooth_y)})")
            print(f"Predicted Target: ({target_x}, {target_y})")
            print(f"\n--- ERRORS ---")
            print(f"Error X: {error_x:+4d} px")
            print(f"Error Y: {error_y:+4d} px")
            print(f"Total Error: {total_error:.1f} px")
            print(f"\n--- VELOCITY ---")
            print(f"Velocity X: {vx:+6.1f} px/s")
            print(f"Velocity Y: {vy:+6.1f} px/s")
            print(f"\n--- CONTROL ---")
            print(f"Should Move: {should_move_motors}")
            print(f"Servo Pan:  {pan:+6.3f} ({pan*100:+6.1f}%)")
            print(f"Servo Tilt: {tilt:+6.3f} ({tilt*100:+6.1f}%)")
            print(f"Gain Used: {current_gain:.4f}")
            print(f"Effective Gain: {effective_gain:.4f} (scaled by error)")
            print(f"Max Step: {current_max_step:.3f}")
            print(f"{'='*60}")
        
        # Draw centering zone
        cv2.rectangle(frame, 
                     (cx_frame - CENTERING_THRESHOLD, cy_frame - CENTERING_THRESHOLD),
                     (cx_frame + CENTERING_THRESHOLD, cy_frame + CENTERING_THRESHOLD),
                     (0, 255, 0), 1)
        
    else:
        target_x, target_y = cx_frame, cy_frame
        error_x = error_y = 0
        tracking_state = STATE_SEARCHING

    # Update servo positions with DEADZONE
    if should_move_motors:
        current_time = time.time()
        dt = current_time - previous_time
        
        # Calculate derivative with TIGHTER clamping
        if GAIN_D > 0 and dt > 0 and dt < 0.5:
            derivative_x = (error_x - previous_error_x) / dt
            derivative_y = (error_y - previous_error_y) / dt
            # Clamp harder to reduce fighting with prediction
            derivative_x = max(-80, min(80, derivative_x))
            derivative_y = max(-80, min(80, derivative_y))
        else:
            derivative_x = derivative_y = 0
        
        previous_error_x = error_x
        previous_error_y = error_y
        previous_time = current_time
        
        # PD control WITH DEADZONE using effective_gain
        if abs(error_x) > DEADZONE:
            p_term_x = error_x * effective_gain  # Use scaled gain
            d_term_x = -derivative_x * GAIN_D
            correction_x = p_term_x + d_term_x
            pan += max(-current_max_step, min(current_max_step, correction_x))
        
        if abs(error_y) > DEADZONE:
            p_term_y = error_y * effective_gain  # Use scaled gain
            d_term_y = -derivative_y * GAIN_D
            correction_y = p_term_y + d_term_y
            tilt += max(-current_max_step, min(current_max_step, correction_y))
    
    # Clamp servo values
    pan = max(-1.0, min(1.0, pan))
    tilt = max(-1.0, min(1.0, tilt))

    # Update servos
    pan_servo.value = pan
    tilt_servo.value = tilt

    # Draw center crosshair
    cv2.line(frame, (cx_frame - 10, cy_frame), (cx_frame + 10, cy_frame), (255, 255, 255), 1)
    cv2.line(frame, (cx_frame, cy_frame - 10), (cx_frame, cy_frame + 10), (255, 255, 255), 1)

    # FPS counter
    frame_count += 1
    if now - fps_start_time > 1.0:
        current_fps = frame_count / (now - fps_start_time)
        frame_count = 0
        fps_start_time = now
    
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    with stream_lock:
        stream_frame = frame.copy()