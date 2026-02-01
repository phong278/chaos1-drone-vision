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
# CRITICAL: Match camera resolution closer to YOLO input
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# We'll center-crop to square for YOLO (reduces interpolation noise)

# ================= YOLOv8 ONNX =================
MODEL_PATH = "yolov8n.onnx"
YOLO_CONF = 0.4
YOLO_INTERVAL = 0.1
PERSON_TIMEOUT = 4.0  # shorter timeout to avoid long-lived boxes when people move away

# YOLO Export Command (for reference):
# yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True
# Expected output shape: [1, 84, 8400] where 84 = [cx, cy, w, h, 80 classes]

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
yolo_input = sess.get_inputs()[0].name
yolo_output_shape = sess.get_outputs()[0].shape  # Store for validation

last_yolo_time = 0
last_person_time = 0
person_box = None


def iou(boxA, boxB):
    # box = (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)
# ================= VELOCITY PREDICTION =================
# Track smoothed position and velocity (not raw YOLO)
smoothed_position = None
smoothed_velocity = (0, 0)
position_smoothing = 0.5   # HIGHER = more responsive, less lag (was 0.15)
velocity_smoothing = 0.6   # HIGHER = faster velocity tracking (was 0.3)
# Shorter latency and miss counters to avoid over-trusting prediction or stale boxes
predicted_latency = 0.08   # System latency estimate (80ms)
missed_detection_count = 0
MISSED_DETECTIONS_THRESHOLD = 2  # YOLO cycles before switching to a new detection
consecutive_no_detections = 0
NO_DETECTION_YOLO_THRESHOLD = 2  # YOLO cycles to clear stale person box
last_detected_pos = None
last_detected_time = None
MIN_GAIN_SCALE = 0.25  # Minimum fraction of gain to avoid collapse near center
# Prediction and centering safety parameters
PRED_OFFSET_MAX_FRAC = 0.25  # Max fraction of measured error allowed for predictive offset (more conservative)
CENTERED_CONFIRM = 3        # Frames required to confirm centered -> LOCKED
centered_counter = 0

def estimate_velocity():
    """Return current smoothed velocity"""
    return smoothed_velocity

def update_position_velocity(detected_x, detected_y, dt):
    """Update smoothed position and velocity using EMA, compute raw velocity from consecutive detections (not from smoothed pos)."""
    global smoothed_position, smoothed_velocity, last_detected_pos, last_detected_time, position_history

    t_now = time.time()
    position_history.append((detected_x, detected_y, t_now))

    # Compute raw velocity from the last two detections to avoid using the lagging smoothed position
    if last_detected_pos is None:
        raw_vx = raw_vy = 0.0
        last_detected_pos = (detected_x, detected_y)
        last_detected_time = t_now
    else:
        dt_raw = t_now - last_detected_time if (t_now - last_detected_time) > 1e-6 else 1e-6
        # If detections are far apart in time, velocity estimate is unreliable
        if dt_raw > 0.4:
            raw_vx = 0.0
            raw_vy = 0.0
            # update last_detected_* so we don't compute huge velocities next cycle
            last_detected_pos = (detected_x, detected_y)
            last_detected_time = t_now
        else:
            raw_vx = (detected_x - last_detected_pos[0]) / dt_raw
            raw_vy = (detected_y - last_detected_pos[1]) / dt_raw
            last_detected_pos = (detected_x, detected_y)
            last_detected_time = t_now
            # Clamp raw velocity to a reasonable maximum to avoid bursts
            max_raw_v = 2000.0
            raw_vx = max(-max_raw_v, min(max_raw_v, raw_vx))
            raw_vy = max(-max_raw_v, min(max_raw_v, raw_vy))

    # Smooth measured position (EMA filter)
    if smoothed_position is None:
        smoothed_position = (detected_x, detected_y)
    else:
        new_x = position_smoothing * detected_x + (1 - position_smoothing) * smoothed_position[0]
        new_y = position_smoothing * detected_y + (1 - position_smoothing) * smoothed_position[1]
        smoothed_position = (new_x, new_y)

    # Smooth velocity using raw detected velocity (EMA) - avoids large time-scaling bugs
    vx_smooth = velocity_smoothing * raw_vx + (1 - velocity_smoothing) * smoothed_velocity[0]
    vy_smooth = velocity_smoothing * raw_vy + (1 - velocity_smoothing) * smoothed_velocity[1]
    # Clamp smoothed velocity to avoid prediction spikes
    vx_smooth = max(-500.0, min(500.0, vx_smooth))
    vy_smooth = max(-500.0, min(500.0, vy_smooth))
    smoothed_velocity = (vx_smooth, vy_smooth) 

def predict_position(current_x, current_y, vx, vy):
    """Predict where target will be after latency"""
    predicted_x = current_x + vx * predicted_latency
    predicted_y = current_y + vy * predicted_latency
    return int(predicted_x), int(predicted_y)

def normalized_error(error, half_range):
    return max(-1.0, min(1.0, error / half_range))
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
DEADZONE = 1  # REDUCED - allows continuous fine adjustments (was 8)

# Thresholds
CENTERING_THRESHOLD = 18
MAX_STEP_PAN = 0.05
MAX_STEP_TILT = 0.05
MAX_STEP_LOCKED_PAN = 0.05
MAX_STEP_LOCKED_TILT = 0.05

# FPS tracking
frame_count = 0
fps_start_time = time.time()
current_fps = 0

# Debug printing
last_print_time = 0
PRINT_INTERVAL = 1.0  # Print debug summary every 1.0 seconds (less noisy)

# Servo safety & tuning
SERVO_MAX_RATE = 0.12  # max servo units per second (slows large jumps)
NO_PROGRESS_THRESHOLD = 6  # frames without error improvement before backing off
no_progress_frames = 0
last_error_mag = None
prev_pan = pan
prev_tilt = tilt

# Stronger damping multiplier for derivative term
DERIV_DAMP_MULT = 0.06  # increased from 0.02 to reduce oscillations

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
    
    # Center crop to square to reduce scaling artifacts
    # This dramatically reduces box jitter
    if w > h:
        # Landscape - crop sides
        crop_x = (w - h) // 2
        frame_crop = frame[:, crop_x:crop_x+h]
    else:
        # Portrait - crop top/bottom
        crop_y = (h - w) // 2
        frame_crop = frame[crop_y:crop_y+w, :]
    
    # Now resize square to square (minimal distortion)
    img = cv2.resize(frame_crop, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    output = sess.run(None, {yolo_input: img})[0]
    output = np.squeeze(output)
    
    # VALIDATION: Check output shape
    if output.shape[0] not in [84, 85]:
        print(f"⚠️ WARNING: Unexpected YOLO output shape: {output.shape}")

    boxes = output[:4, :]  # [cx, cy, w, h]
    scores = output[4:, :]  # [80 class scores]

    class_ids = np.argmax(scores, axis=0)
    confidences = scores[class_ids, np.arange(scores.shape[1])]

    # Collect ALL person detections (class 0 = person in COCO)
    detections = []
    crop_size = min(h, w)
    
    for i in range(scores.shape[1]):
        if class_ids[i] == 0 and confidences[i] > YOLO_CONF:
            cx, cy, bw, bh = boxes[:, i]
            # Map back from cropped square to original frame
            x1_crop = int((cx - bw / 2) * crop_size)
            y1_crop = int((cy - bh / 2) * crop_size)
            x2_crop = int((cx + bw / 2) * crop_size)
            y2_crop = int((cy + bh / 2) * crop_size)
            
            # Adjust back to original frame coordinates
            if w > h:
                x1 = x1_crop + (w - h) // 2
                y1 = y1_crop
                x2 = x2_crop + (w - h) // 2
                y2 = y2_crop
            else:
                x1 = x1_crop
                y1 = y1_crop + (h - w) // 2
                x2 = x2_crop
                y2 = y2_crop + (h - w) // 2
            
            # PROFESSIONAL TRICK: Track chest/torso, not box center
            # This is 35% down from top = chest level
            # Much more stable than geometric center
            center_x = (x1 + x2) // 2
            center_y = y1 + int(0.35 * (y2 - y1))  # Chest point, not center
            
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

        # Determine current target center if we already have a person tracked
        current_target_pos = None
        if person_box is not None:
            x1, y1, x2, y2, conf = person_box
            current_target_pos = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Select best person to track
        best_person = select_best_person(detections, current_target_pos)

        if best_person:
            x1, y1, x2, y2 = best_person['box']
            conf = best_person['conf']
            new_box = (x1, y1, x2, y2)

            if person_box is not None:
                old_box = person_box[:4]
                overlap = iou(old_box, new_box)

                # Distance between centers
                old_cx = (old_box[0] + old_box[2]) // 2
                old_cy = (old_box[1] + old_box[3]) // 2
                new_cx = (x1 + x2) // 2
                new_cy = (y1 + y2) // 2
                dist = np.hypot(new_cx - old_cx, new_cy - old_cy)

                # Accept if the overlap is reasonable OR the detection is very close
                if overlap >= 0.25 or dist < 80:
                    person_box = (x1, y1, x2, y2, conf)
                    last_person_time = now
                    missed_detection_count = 0
                    consecutive_no_detections = 0
                else:
                    # If we see a consistently different detection, switch to it after a couple cycles
                    missed_detection_count += 1
                    if missed_detection_count > MISSED_DETECTIONS_THRESHOLD:
                        person_box = (x1, y1, x2, y2, conf)
                        last_person_time = now
                        missed_detection_count = 0
            else:
                person_box = (x1, y1, x2, y2, conf)
                last_person_time = now
                missed_detection_count = 0
                consecutive_no_detections = 0
        else:
            # No valid person detected; increment a counter and clear stale boxes quickly
            consecutive_no_detections += 1
            if consecutive_no_detections > NO_DETECTION_YOLO_THRESHOLD:
                person_box = None
                missed_detection_count = 0


    # Handle timeout - clear stale boxes faster and decay filters more predictably
    if last_person_time and (now - last_person_time > PERSON_TIMEOUT):
        person_box = None
        tracking_state = STATE_SEARCHING
        position_history.clear()

        # Gradually decay velocity to avoid lunges on reacquisition
        if smoothed_velocity != (0, 0):
            vx, vy = smoothed_velocity
            smoothed_velocity = (vx * 0.8, vy * 0.8)

        # Reset smoothed_position only after a long absence
        if last_person_time > 0 and now - last_person_time > 3.0:
            smoothed_position = None
            smoothed_velocity = (0, 0)

        last_position = None
        # NOTE: pan/tilt NOT reset - holds last position

    # Process tracking with PREDICTION
    should_move_motors = False
    tracked_center = None  # Store center of tracked person
    current_gain = 0  # Initialize here so it's always defined
    current_max_step_pan = 0
    current_max_step_tilt = 0
    effective_gain = 0
    
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
        
        # MEASURED state and velocity
        vx, vy = estimate_velocity()
        smooth_x, smooth_y = (smoothed_position if smoothed_position else (detected_x, detected_y))

        # Measured error (based on measurement, not prediction) used for PD
        measured_error_x = int(round(smooth_x)) - cx_frame
        measured_error_y = int(round(smooth_y)) - cy_frame

        # Prediction component: scale with velocity and recency so we don't trust it for long
        vel_mag = np.hypot(vx, vy)
        time_since_seen = now - (last_position_time if last_position_time else now)
        # Be conservative: scale velocity more and cap trust lower to avoid prediction-driven overshoot
        pred_trust = np.exp(-time_since_seen / 0.5) * min(1.0, vel_mag / 300.0)
        pred_trust = max(0.0, min(0.6, pred_trust))

        pred_offset_x = vx * predicted_latency * pred_trust
        pred_offset_y = vy * predicted_latency * pred_trust

        # Limit predictive offset so it cannot flip the sign of the measured error
        if abs(measured_error_x) < 2:
            pred_offset_x = 0.0
        else:
            pred_offset_x = np.sign(measured_error_x) * min(abs(pred_offset_x), abs(measured_error_x) * PRED_OFFSET_MAX_FRAC)

        if abs(measured_error_y) < 2:
            pred_offset_y = 0.0
        else:
            pred_offset_y = np.sign(measured_error_y) * min(abs(pred_offset_y), abs(measured_error_y) * PRED_OFFSET_MAX_FRAC)

        # Final error used for driving motors includes a small predictive offset
        final_error_x = measured_error_x + pred_offset_x
        final_error_y = measured_error_y + pred_offset_y

        # Target for visualization (clamped)
        target_x = max(0, min(w, int(round(cx_frame + final_error_x))))
        target_y = max(0, min(h, int(round(cy_frame + final_error_y))))

        # If within the very small deadzone, don't move motors to avoid jitter
        in_deadzone = (abs(measured_error_x) < DEADZONE and abs(measured_error_y) < DEADZONE)
        if in_deadzone:
            final_error_x = 0
            final_error_y = 0
            should_move_motors = False
            previous_error_x = measured_error_x
            previous_error_y = measured_error_y

        # State machine uses MEASURED error (do not get locked based on prediction)
        if tracking_state == STATE_SEARCHING:
            tracking_state = STATE_CENTERING
            should_move_motors = True
            status_text = "CENTERING"
            status_color = (0, 165, 255)
            
        elif tracking_state == STATE_CENTERING:
            should_move_motors = True
            # Use measured error with hysteresis to decide if centered
            if is_centered(measured_error_x, measured_error_y):
                centered_counter += 1
                if centered_counter >= CENTERED_CONFIRM:
                    tracking_state = STATE_LOCKED
                    lock_time = now
                    status_text = "LOCKED"
                    status_color = (0, 255, 0)
                else:
                    status_text = f"CENTERING ({centered_counter})"
                    status_color = (0, 165, 255)
            else:
                centered_counter = 0
                status_text = "CENTERING"
                status_color = (0, 165, 255)
                
        elif tracking_state == STATE_LOCKED:
            # ALWAYS track when locked - no more "steady" state
            should_move_motors = True
            status_text = "LOCKED - TRACKING"
            status_color = (0, 255, 0)  # Green when locked

        # === SELECT GAIN BEFORE DEBUG PRINT ===
        if tracking_state == STATE_CENTERING:
            current_gain = GAIN_CENTERING
            current_max_step_pan = MAX_STEP_PAN
            current_max_step_tilt = MAX_STEP_TILT
        elif tracking_state == STATE_LOCKED:
            current_gain = GAIN_LOCKED
            current_max_step_pan = MAX_STEP_LOCKED_PAN
            current_max_step_tilt = MAX_STEP_LOCKED_TILT
        else:
            current_gain = GAIN_TRACKING
            current_max_step_pan = MAX_STEP_PAN
            current_max_step_tilt = MAX_STEP_TILT

        # Distance-based gain scaling AND adaptive step size
        error_mag_meas = np.hypot(measured_error_x, measured_error_y)
        gain_scale = min(1.0, error_mag_meas / 80.0)
        # Prevent gain collapsing to zero near center by enforcing a minimum scale
        effective_gain = current_gain * (MIN_GAIN_SCALE + (1.0 - MIN_GAIN_SCALE) * gain_scale)
        
        # SMART: Increase authority near center but more conservatively
        if error_mag_meas < 40:
            current_max_step_pan *= 1.10
            current_max_step_tilt *= 1.10

        # Track progress: if error doesn't improve for several frames, back off to avoid oscillation
        total_error = error_mag_meas
        if last_error_mag is not None:
            if total_error >= last_error_mag - 1.0:
                no_progress_frames += 1
            else:
                no_progress_frames = 0
        last_error_mag = total_error
        
     
        
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
        
        # Draw predicted center (green) - where we are compensating for latency
        if smoothed_position and (abs(vx) > 2 or abs(vy) > 2):
            cv2.circle(frame, (target_x, target_y), 5, (0, 255, 0), -1)
            cv2.line(frame, (int(round(smooth_x)), int(round(smooth_y))), (target_x, target_y), (0, 255, 0), 2)
            cv2.putText(frame, f"PREDICT (trust={pred_trust:.2f})", (target_x + 8, target_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Draw error visualization - line from target to frame center
        cv2.line(frame, (target_x, target_y), (cx_frame, cy_frame), (255, 0, 255), 2)
        cv2.putText(frame, f"Err(meas/final): {int(abs(measured_error_x))}/{int(abs(final_error_x))} px, {int(abs(measured_error_y))}/{int(abs(final_error_y))} px", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # === DEBUG PRINTING ===
        if now - last_print_time > PRINT_INTERVAL:
            last_print_time = now

            total_error_meas = np.hypot(measured_error_x, measured_error_y)
            total_error_final = np.hypot(final_error_x, final_error_y)

            print(f"\n{'='*60}")
            print(f"TRACKING SUMMARY @ {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            print(f"State: {['SEARCHING','CENTERING','LOCKED'][tracking_state]}  centered={centered_counter}  FPS={current_fps:.1f}Hz  YOLO={YOLO_INTERVAL:.2f}s")

            if person_box:
                print(f"Target: box={x1,y1,x2,y2} conf={conf:.2f}  Raw=({detected_x},{detected_y})  Smooth=({int(smooth_x)},{int(smooth_y)})")
                print(f"Err(meas/final): x={measured_error_x:+4d}/{final_error_x:+6.1f} px  y={measured_error_y:+4d}/{final_error_y:+6.1f} px  total_meas={total_error_meas:.1f} total_final={total_error_final:.1f}")
                print(f"Pred: vx={vx:+6.1f} vy={vy:+6.1f} px/s  pred_trust={pred_trust:.2f}  pred_off=({pred_offset_x:+5.1f},{pred_offset_y:+5.1f})")
                # Compute a local center boost for logging (may not be defined yet in servo update)
                nx_local = normalized_error(final_error_x, cx_frame)
                ny_local = normalized_error(final_error_y, cx_frame)
                center_boost_local = 1.0 + 0.4 * (1.0 - min(1.0, abs(nx_local) + abs(ny_local)))
                print(f"Control: gain={current_gain:.4f} eff_gain={effective_gain:.4f} center_boost={center_boost_local:.3f}  steps=(pan={current_max_step_pan:.3f},tilt={current_max_step_tilt:.3f})")
                print(f"Servo: pan={pan:+6.3f} tilt={tilt:+6.3f}  should_move={should_move_motors}")
            else:
                print("No current target")

            print(f"Detections: missed={missed_detection_count} consecutive_no={consecutive_no_detections}  PERSON_TIMEOUT={PERSON_TIMEOUT}s")

            # Helpful notes
            if total_error_final < CENTERING_THRESHOLD:
                print("NOTE: within centering threshold")
            if pred_trust > 0.25:
                print("NOTE: prediction contributing significantly — consider reducing latency/trust if unstable")
            print(f"{'='*60}")
            print(f"\n--- CONTROL ---")
            print(f"Should Move: {should_move_motors}")
            print(f"Servo Pan:  {pan:+6.3f} ({pan*100:+6.1f}%)")
            print(f"Servo Tilt: {tilt:+6.3f} ({tilt*100:+6.1f}%)")
            print(f"Gain Used: {current_gain:.4f}")
            print(f"Effective Gain: {effective_gain:.4f} (scaled by error)")
            print(f"Max Step Pan: {current_max_step_pan:.3f}")
            print(f"Max Step Tilt: {current_max_step_tilt:.3f}")
            print(f"{'='*60}")
        
        # Draw centering zone
        cv2.rectangle(frame, 
                     (cx_frame - CENTERING_THRESHOLD, cy_frame - CENTERING_THRESHOLD),
                     (cx_frame + CENTERING_THRESHOLD, cy_frame + CENTERING_THRESHOLD),
                     (0, 255, 0), 1)
        
    else:
        target_x, target_y = cx_frame, cy_frame
        # No target - zero errors and prediction influence
        measured_error_x = 0
        measured_error_y = 0
        final_error_x = 0
        final_error_y = 0
        pred_trust = 0.0
        vx = vy = 0.0
        tracking_state = STATE_SEARCHING
        centered_counter = 0
        missed_detection_count = 0
        consecutive_no_detections = 0

    # Update servo positions with DEADZONE
    if should_move_motors:
        current_time = time.time()
        dt = current_time - previous_time
        
        # Calculate derivative using measured error (not prediction)
        if GAIN_D > 0 and dt > 0 and dt < 0.5:
            derivative_x = (measured_error_x - previous_error_x) / dt
            derivative_y = (measured_error_y - previous_error_y) / dt
            # Clamp harder to reduce fighting with prediction
            derivative_x = max(-80, min(80, derivative_x))
            derivative_y = max(-80, min(80, derivative_y))
        else:
            derivative_x = derivative_y = 0
        
        # Update previous error/time based on measured values
        previous_error_x = measured_error_x
        previous_error_y = measured_error_y
        previous_time = current_time
        
        # Use final error (measurement + predictive offset) for desired direction
        nx = normalized_error(final_error_x, cx_frame)
        ny = normalized_error(final_error_y, cy_frame)

        # Boost authority near center more conservatively to avoid collapse/overshoot
        center_boost = 1.0 + 0.4 * (1.0 - min(1.0, abs(nx) + abs(ny)))

        # Compute magnitude and unit direction along the purple line (from target to center)
        mag = np.hypot(nx, ny)
        if mag > 1e-6:
            ux = nx / mag
            uy = ny / mag

                # Desired magnitude scales with pixel error; 160px -> full step
            desired_magnitude = min(1.0, error_mag_meas / 160.0)
            base_step = max(current_max_step_pan, current_max_step_tilt)

            # Scaled max per-axis step (more conservative caps)
            scaled_max_pan = min(0.12, current_max_step_pan + 0.0005 * error_mag_meas)
            scaled_max_tilt = min(0.12, current_max_step_tilt + 0.0005 * error_mag_meas)

            # If error is large and target is relatively static, use a slightly larger raw move (but conservative)
            vel_mag = np.hypot(vx, vy)
            if error_mag_meas >= 120 and vel_mag < 30:
                raw_move = min(0.12, 0.01 + 0.0009 * error_mag_meas)
            else:
                # Ensure some responsiveness — small errors still move at a fraction of base step
                raw_move = base_step * max(0.12, desired_magnitude)

            # Reduce move if prediction trust is high (avoid chasing predictive spikes)
            if pred_trust > 0.35:
                raw_move *= 0.5

            # If we've had no progress recently, back off for stability
            if no_progress_frames > NO_PROGRESS_THRESHOLD:
                raw_move *= 0.35
                scaled_max_pan *= 0.5
                scaled_max_tilt *= 0.5

            # Project derivative onto the same direction for damping
            deriv_proj = (derivative_x * ux + derivative_y * uy)
            damp = deriv_proj * GAIN_D * DERIV_DAMP_MULT

            pan_delta = ux * raw_move - ux * damp
            tilt_delta = uy * raw_move - uy * damp

            # Limit per-axis deltas by both scaled step and servo rate limit (dt)
            allowed_change = SERVO_MAX_RATE * dt
            pan_limit = min(scaled_max_pan, allowed_change)
            tilt_limit = min(scaled_max_tilt, allowed_change)

            pan_delta = np.clip(pan_delta, -pan_limit, pan_limit)
            tilt_delta = np.clip(tilt_delta, -tilt_limit, tilt_limit)

            # Apply pan (x) and tilt (y). Apply tilt with positive error increasing tilt (move camera down)
            # Safety: if servo near limit, do not push further
            near_limit = False
            if (pan + pan_delta) > 0.95 or (pan + pan_delta) < -0.95:
                pan_delta = 0.0
                near_limit = True
            if (tilt + tilt_delta) > 0.95 or (tilt + tilt_delta) < -0.95:
                tilt_delta = 0.0
                near_limit = True

            pan += pan_delta
            tilt += tilt_delta

            if near_limit:
                # If we hit a physical limit, reduce motion for several frames to avoid oscillation
                missed_detection_count = 0
                consecutive_no_detections = 0

            # Update no-progress: if motion but no improvement, increment counter handled earlier
            # Track previous pan/tilt for rate or stuck detection
            prev_moved = (abs(prev_pan - pan) > 1e-4) or (abs(prev_tilt - tilt) > 1e-4)
            if prev_moved and total_error >= last_error_mag - 0.5:
                no_progress_frames += 1
            else:
                # if we did move and saw improvement, reset counter
                if total_error < last_error_mag - 1.0:
                    no_progress_frames = 0

            # Update previous servo values for next iteration
            prev_pan = pan
            prev_tilt = tilt
        else:
            # No significant direction - no proportional move
            pass



    
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
        if frame is not None:
            stream_frame = frame.copy()