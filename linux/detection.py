#!/usr/bin/env python3
import cv2
import time
import threading
from ultralytics import YOLO
import RPi.GPIO as GPIO  # or pigpio, depending on your setup
import math

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except:
        return None

class SimpleStreamer:
    def __init__(self, port=5000):
        self.port = port
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def update_frame(self, frame):
        with self.frame_lock:
            self.frame = frame
    
    def _generate_frame(self):
        while self.running:
            with self.frame_lock:
                if self.frame is not None:
                    _, jpeg = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if _:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               jpeg.tobytes() + b'\r\n')
            time.sleep(0.033)  
    
    def run(self):
        try:
          
            from http.server import HTTPServer, BaseHTTPRequestHandler
            
            class StreamingHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/stream.mjpg':
                        self.send_response(200)
                        self.send_header('Content-Type', 
                                       'multipart/x-mixed-replace; boundary=frame')
                        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                        self.end_headers()
                        
                        for frame in self.server.streamer._generate_frame():
                            try:
                                self.wfile.write(frame)
                            except:
                                break
                    elif self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b'''
                        <html><head><title>Camera Stream</title></head>
                        <body style="margin:0;background:#000;">
                            <img src="/stream.mjpg">
                        </body></html>
                        ''')
            
            class StreamingServer(HTTPServer):
                def __init__(self, *args, **kwargs):
                    self.streamer = kwargs.pop('streamer')
                    super().__init__(*args, **kwargs)
            
            self.running = True
            server = StreamingServer(('0.0.0.0', self.port), StreamingHandler, streamer=self)
            print(f"Streaming at http://localhost:{self.port}")
            print(f"Network: http://{self._get_ip()}:{self.port}")
            server.serve_forever()
            
        except Exception as e:
            print(f"Streaming error: {e}")
    
    def _get_ip(self):
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))
            return s.getsockname()[0]
        except:
            return "YOUR_PI_IP"
        
    def stop(self):
        self.running = False

# Core Detection System
class DetectionSystem:
    def __init__(self, config):
        self.config = config
        self.model = YOLO('models/openvino/')
        self.model.conf = config['detection']['confidence']
        # Initialize camera
        self.cap = cv2.VideoCapture(
            config['camera']['device_id'],
            cv2.CAP_V4L2
        )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
        self.cap.set(cv2.CAP_PROP_FPS, config['camera']['fps'])
        
        #----GIMBAL-----#
        self.pan_pin = 17   # GPIO pin for pan motor
        self.tilt_pin = 27  # GPIO pin for tilt motor
        self.setup_motors()
        self.pan_angle = 90   # start at neutral
        self.tilt_angle = 90  # start at neutral
        self.max_angle = 180
        self.min_angle = 0
        self.kp = 0.1  # proportional gain
        #---------------#
        
        self.streamer = None
        if config['system']['enable_streaming']:
            self.streamer = SimpleStreamer(port=config['system']['streaming_port'])
            threading.Thread(target=self.streamer.run, daemon=True).start()
            time.sleep(1)  

    def setup_motors(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pan_pin, GPIO.OUT)
        GPIO.setup(self.tilt_pin, GPIO.OUT)
        self.pan_pwm = GPIO.PWM(self.pan_pin, 50)
        self.tilt_pwm = GPIO.PWM(self.tilt_pin, 50)
        self.pan_pwm.start(7.5)   # neutral
        self.tilt_pwm.start(7.5)  # neutral

    def move_motors(self, x_offset, y_offset):
        # P-controller
        self.pan_angle += self.kp * x_offset
        self.tilt_angle -= self.kp * y_offset  # usually inverted

        # clamp angles
        self.pan_angle = max(self.min_angle, min(self.max_angle, self.pan_angle))
        self.tilt_angle = max(self.min_angle, min(self.max_angle, self.tilt_angle))

        # convert angle to duty cycle for standard servo
        def angle_to_duty(a):
            return 2.5 + (a / 180.0) * 10.0

        self.pan_pwm.ChangeDutyCycle(angle_to_duty(self.pan_angle))
        self.tilt_pwm.ChangeDutyCycle(angle_to_duty(self.tilt_angle))

    def track_object(self, detections, frame):
        # Simple: take the first detection as target
        if not detections:
            return
        det = detections[0]
        x1, y1, x2, y2 = det['bbox']
        obj_x = (x1 + x2) / 2
        obj_y = (y1 + y2) / 2

        frame_cx = frame.shape[1] / 2
        frame_cy = frame.shape[0] / 2

        x_offset = obj_x - frame_cx
        y_offset = obj_y - frame_cy

        self.move_motors(x_offset, y_offset)

    def detect(self, frame):
        """Run YOLOv8 detection on frame"""
        inference_size = self.config['performance']['inference_size']
        results = self.model(frame, imgsz=inference_size, conf=self.config['detection']['confidence'], verbose=False)
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < self.config['detection']['confidence']:
                    continue
                class_id = int(box.cls[0])
                label = self.model.names[class_id] if class_id < len(self.model.names) else 'object'
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': box.xyxy[0].cpu().numpy().astype(int)
                })           
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            
            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label}: {conf:.1%}"
            cv2.putText(frame, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
    def run(self):
        print("Starting Object Detection")
        print(f"Camera: {self.config['camera']['width']}x{self.config['camera']['height']}")
        print(f"Confidence: {self.config['detection']['confidence']}")
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        last_detections = []
        try:
            while True:
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera error")
                    break
                frame_count += 1
                # Frame skipping
                if frame_count % (self.config['performance']['frame_skip'] + 1) == 0:
                    last_detections = self.detect(frame)
                detections = last_detections
                self.track_object(detections, frame)
                                           
                # Draw detections
                display_frame = self.draw_detections(frame.copy(), detections)
                # Add FPS overlay
                if time.time() - fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    fps_time = time.time()
                    if detections:
                        print(f"{fps} FPS | Detected: {len(detections)} objects")
                
                cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Stream frame
                if self.streamer:
                    self.streamer.update_frame(display_frame)
                # Throttle to target FPS
                elapsed = time.time() - start_time
                target = 1.0 / self.config['performance']['throttle_fps']
                if elapsed < target:
                    time.sleep(target - elapsed)
             
        except KeyboardInterrupt:
            print("\n Stopping...")
        finally:
            self.cap.release()
            if self.streamer:
                self.streamer.stop()

if __name__ == "__main__":
    # Load config 
    config = {
        'system': {
            'enable_streaming': True,
            'streaming_port': 5000,
            'max_streaming_clients': 5
        },
        'performance': {
            'throttle_fps': 15,
            'frame_skip': 2,
            'inference_size': 640,
            'use_threading': True,
        },
        'camera': {
            'device_id': 0,
            'width': 1280,
            'height': 720,
            'fps': 15,
            
        },
        'detection': {
            'model_cfg': 'yolov8n.pt',
            'confidence': 0.4
        }
    }
    
    detector = DetectionSystem(config)
    detector.run()