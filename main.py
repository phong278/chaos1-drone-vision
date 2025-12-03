import cv2
import sys
import signal
import time
import os
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# ==================== CONFIGURATION ====================
CONFIG = {
    "system": {
        "platform": "auto",  # "auto", "windows", "linux", "raspberry"
        "log_to_file": True,
        "data_folder": "detection_data"
    },
    "performance": {
        "mode": "balanced",  # "balanced", "high_performance"
        "throttle_fps": 30,  # Target FPS, will add sleep to not exceed this
        "frame_skip": 0,  # Number of frames to skip between processing
    },
    "camera": {
        "device_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "flip_horizontal": False,
        "flip_vertical": False,
    },
    "detection": {
        "model": "yolov8n.pt",
        "confidence": 0.5,
        "iou_threshold": 0.5,
        "track_objects": True,  # Enable object tracking between frames
        # Expanded object classes to detect
        "classes": {
            "electronics": ["cell phone", "laptop", "mouse", "keyboard", "remote"],
            "people": ["person"],
            "vehicles": ["car", "motorcycle", "bicycle", "bus", "truck"],
            "animals": ["cat", "dog", "bird"],
            "furniture": ["chair", "dining table", "bed", "couch"],
            "kitchen": ["bottle", "cup", "bowl", "fork", "knife", "spoon"],
            "sports": ["sports ball", "baseball bat", "baseball glove", 
                      "tennis racket", "skateboard"],
        }
    },
    "tracking": {
        "priority_order": ["cell phone", "person", "laptop", "cat", "dog"],
        "auto_focus": True,  # Automatically focus on highest priority object
        "deadzone": 50,
        "history_length": 10,  # Number of past positions to track
    },
    "visualization": {
        "show_fps": True,
        "show_center": True,
        "show_bbox": True,
        "show_trail": True,  # Show movement trail
        "show_distance": False,  # Estimate distance (requires calibration)
        "color_scheme": "vibrant",  # "vibrant", "pastel", "mono"
        "window_scale": 1.0,  # Scale window size
    },
    "output": {
        "console_log": False,
        "file_log": False,
        "save_detections": True,  # Save detection images
        "save_interval": 30,  # Save every 30 seconds
    }
}

# ==================== COLOR SCHEMES ====================
COLOR_SCHEMES = {
    "vibrant": {
        "cell phone": (0, 255, 0),      # Green
        "person": (255, 0, 0),          # Blue
        "laptop": (255, 255, 0),        # Cyan
        "cat": (255, 0, 255),           # Magenta
        "dog": (0, 255, 255),           # Yellow
        "car": (255, 165, 0),           # Orange
        "bottle": (128, 0, 128),        # Purple
        "chair": (0, 128, 128),         # Teal
        "default": (200, 200, 200)      # Gray
    },
    "pastel": {
        "cell phone": (100, 230, 100),  # Light Green
        "person": (100, 100, 230),      # Light Blue
        "default": (180, 180, 180)      # Light Gray
    }
}

# ==================== INITIALIZATION ====================

class DetectionSystem:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.cap = None
        self.detection_history = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.focus_object = None
        self.platform = self.detect_platform()
        
        # Performance mode adjustments
        self.is_high_performance = self.config["performance"]["mode"] == "high_performance"
        if self.is_high_performance:
            self.apply_high_performance_settings()

        # Initialize logging
        self.setup_logging()
        
        # Initialize system
        self.init_system()
    
    def apply_high_performance_settings(self):
        """Overrides config for high-performance mode."""
        # Disable all visualization
        for key in self.config["visualization"]:
            self.config["visualization"][key] = False
        
        # Disable file output
        self.config["output"]["save_detections"] = False
        self.config["output"]["file_log"] = False
        
        # Disable tracking features that are not essential
        self.config["detection"]["track_objects"] = False

    def detect_platform(self):
        """Detect the current platform"""
        import platform
        system = platform.system().lower()
        
        if system == "linux":
            # Check if Raspberry Pi
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    if 'raspberry' in f.read().lower():
                        return "raspberry"
            except:
                pass
            return "linux"
        elif system == "windows":
            return "windows"
        elif system == "darwin":
            return "mac"
        return "unknown"
    
    def setup_logging(self):
        """Setup logging system"""
        if self.config["system"]["log_to_file"]:
            # Create data folder
            data_folder = self.config["system"]["data_folder"]
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(data_folder, f"detection_log_{timestamp}.txt")
            
            with open(self.log_file, 'w') as f:
                f.write(f"Detection System Log - Started at {timestamp}\n")
                f.write(f"Platform: {self.platform}\n")
                f.write(f"Performance Mode: {self.config['performance']['mode']}\n")
                f.write("=" * 50 + "\n")
    
    def log_message(self, message, level="INFO"):
        """Log message to console and/or file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        if self.config["output"]["console_log"]:
            print(formatted_msg)
        
        if self.config["system"]["log_to_file"] and self.config["output"]["file_log"] and hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as f:
                f.write(formatted_msg + "\n")
    
    def init_system(self):
        """Initialize the detection system"""
        self.log_message(f"Initializing on platform: {self.platform}")
        
        # Load model
        self.log_message(f"Loading model: {self.config['detection']['model']}")
        try:
            self.model = YOLO(self.config['detection']['model'])
            self.log_message("Model loaded successfully")
        except Exception as e:
            self.log_message(f"Error loading model: {e}", "ERROR")
            sys.exit(1)
        
        # Initialize camera
        self.init_camera()
    
    def init_camera(self):
        """Initialize camera"""
        self.log_message(f"Initializing camera {self.config['camera']['device_id']}")
        self.cap = cv2.VideoCapture(self.config['camera']['device_id'])
        
        if not self.cap.isOpened():
            self.log_message("Error: Could not open camera", "ERROR")
            sys.exit(1)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        # Get actual properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        self.log_message(f"Camera: {self.frame_width}x{self.frame_height} @ {self.frame_fps:.1f} FPS")
    
    def get_color(self, label):
        """Get color for a label based on color scheme"""
        scheme = COLOR_SCHEMES.get(self.config['visualization']['color_scheme'], COLOR_SCHEMES['vibrant'])
        return scheme.get(label, scheme['default'])
    
    def calculate_priority(self, label, confidence):
        """Calculate priority score for an object"""
        priority_order = self.config['tracking']['priority_order']
        
        if label in priority_order:
            base_priority = len(priority_order) - priority_order.index(label)
        else:
            base_priority = 1
        
        # Adjust by confidence
        return base_priority * confidence
    
    def estimate_distance(self, box_height, known_height=15):
        """Estimate distance to object (in cm, requires calibration)"""
        focal_length = 500  # Example, calibrate this!
        
        if box_height > 0:
            distance = (known_height * focal_length) / box_height
            return distance
        return None
    
    def save_detection_image(self, frame, detections):
        """Save detection image with metadata"""
        if not self.config['output']['save_detections']:
            return
        
        current_time = time.time()
        if current_time - self.last_save_time >= self.config['output']['save_interval']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.config['system']['data_folder'],
                f"detection_{timestamp}.jpg"
            )
            
            if not self.is_high_performance:
                cv2.putText(frame, timestamp, (10, self.frame_height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imwrite(filename, frame)
            self.log_message(f"Saved detection image: {filename}")
            self.last_save_time = current_time
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Flip frame if configured
        if self.config['camera']['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        if self.config['camera']['flip_vertical']:
            frame = cv2.flip(frame, 0)
        
        # Run inference
        results = self.model(frame, 
                           conf=self.config['detection']['confidence'],
                           iou=self.config['detection']['iou_threshold'],
                           verbose=False)
        
        current_detections = {}
        best_priority = 0
        best_object = None
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                found = False
                for category in self.config['detection']['classes'].values():
                    if label in category:
                        found = True
                        break
                
                if not found:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0]
                obj_cx = int((x1 + x2) / 2)
                obj_cy = int((y1 + y2) / 2)
                
                priority = self.calculate_priority(label, conf)
                
                detection_data = {
                    'label': label,
                    'confidence': conf,
                    'center': (obj_cx, obj_cy),
                    'bbox': (x1, y1, x2, y2),
                    'offset': (obj_cx - self.center_x, obj_cy - self.center_y),
                    'priority': priority
                }
                
                if self.config['detection']['track_objects']:
                    if label not in self.detection_history:
                        self.detection_history[label] = []
                    self.detection_history[label].append(detection_data)
                    if len(self.detection_history[label]) > self.config['tracking']['history_length']:
                        self.detection_history[label] = self.detection_history[label][-self.config['tracking']['history_length']:]
                
                current_detections[label] = detection_data
                
                if priority > best_priority:
                    best_priority = priority
                    best_object = detection_data
                
                if not self.is_high_performance:
                    self.draw_detection(frame, detection_data)
        
        if self.config['tracking']['auto_focus'] and best_object:
            self.focus_object = best_object
            pan_angle, tilt_angle = self.control_servo(
                best_object['offset'][0], 
                best_object['offset'][1]
            )

            # -------- SIMPLE TERMINAL OUTPUT FOR BEST OBJECT --------
            if best_object:
                dx, dy = best_object['offset']
                label = best_object['label']
                print(f"BEST_OBJ  dx={dx}  dy={dy}  label={label}")
            # --------------------------------------------------------

            
            if self.frame_count % 30 == 0:
                self.log_message(
                    f"Focus: {best_object['label']} "
                    f"Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°"
                )
        
        if not self.is_high_performance:
            self.draw_overlay(frame, current_detections)
        
        self.save_detection_image(frame, current_detections)
        
        return frame, current_detections
    
    def draw_detection(self, frame, detection):
        """Draw a single detection on frame"""
        label = detection['label']
        conf = detection['confidence']
        x1, y1, x2, y2 = detection['bbox']
        color = self.get_color(label)
        
        if self.config['visualization']['show_bbox']:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        label_text = f"{label} {conf:.2f}"
        cv2.putText(frame, label_text, 
                   (int(x1), int(y1) - 10 if y1 > 20 else int(y1) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_overlay(self, frame, detections):
        """Draw overlay information"""
        if self.config['visualization']['show_fps'] and hasattr(self, 'fps'):
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        obj_count = len(detections)
        count_text = f"Objects: {obj_count}"
        cv2.putText(frame, count_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def run(self):
        """Main run loop"""
        self.log_message("Starting detection system...")
        self.log_message("Press 'q' in the OpenCV window to quit.")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        frame_skip_counter = 0
        
        target_frame_time = 1.0 / self.config['performance']['throttle_fps'] if self.config['performance']['throttle_fps'] > 0 else 0

        while True:
            frame_start_time = time.time()

            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("Error reading frame or end of stream", "ERROR")
                break
            
            self.frame_count += 1
            
            # Frame skipping logic
            if frame_skip_counter > 0:
                frame_skip_counter -= 1
                # Still need to handle window events and delay
                if not self.is_high_performance:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(target_frame_time / 2) # Sleep even on skipped frames
                continue

            frame_skip_counter = self.config['performance']['frame_skip']

            # Process frame
            processed_frame, detections = self.process_frame(frame)
            
            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                self.fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Display frame if not in high performance mode
            if not self.is_high_performance:
                window_name = "Multi-Object Detection System"
                cv2.imshow(window_name, processed_frame)
            
            # Handle keyboard input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.log_message("Quit requested by user")
                break

            # Throttle to target FPS
            elapsed_time = time.time() - frame_start_time
            sleep_time = target_frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.log_message("Shutting down...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        runtime = time.time() - self.start_time
        self.log_message(f"System ran for {runtime:.1f} seconds")
        self.log_message(f"Processed {self.frame_count} frames")

# ==================== MAIN EXECUTION ====================

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n[!] Ctrl+C detected. Shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        system = DetectionSystem(CONFIG)
        system.run()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDetection system terminated.")
