import cv2
import sys
import signal
import time
import os
from datetime import datetime
from pathlib import Path
from collections import deque
from threading import Thread, Lock, Event
from queue import Queue, Full
import numpy as np

# Try to import TensorFlow Lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ==================== CONFIGURATION ====================
CONFIG = {
    "system": {
        "platform": "auto",
        "log_to_file": True,
        "data_folder": "detection_data",
        "max_storage_mb": 500,
        "thermal_throttle": True,
        "temp_threshold": 70,  # Lower for Pi 3
    },
    "performance": {
        "mode": "power_saving",  # Pi 3 needs power saving
        "throttle_fps": 8,  # Lower FPS for Pi 3
        "frame_skip": 2,  # Skip more frames
        "resize_input": True,
        "input_size": (300, 300),  # MobileNet SSD uses 300x300
        "use_threading": False,  # Disable threading on Pi 3 (GIL overhead)
        "frame_buffer_size": 1,
    },
    "camera": {
        "device_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "flip_horizontal": False,
        "flip_vertical": False,
        "backend": cv2.CAP_V4L2,
        "buffer_size": 1,
    },
    "detection": {
        "model": "detect.tflite",  # TFLite model
        "labels": "labelmap.txt",  # COCO labels
        "confidence": 0.5,
        "track_objects": False,  # Disable tracking for performance
        "priority_classes": {
            "high": ["person", "cell phone", "laptop", "cat", "dog"],
            "medium": ["car", "motorcycle", "bicycle", "bus", "truck", "bottle", "cup"],
            "low": ["chair", "couch", "bed", "dining table", "tv", "keyboard", "mouse"]
        },
    },
    "tracking": {
        "priority_order": ["person", "cell phone", "laptop", "cat", "dog"],
        "auto_focus": False,  # Disable for performance
        "history_length": 3,
    },
    "visualization": {
        "show_fps": True,
        "show_bbox": True,
        "show_center": False,
        "color_scheme": "vibrant",
        "render_every_n": 1,
    },
    "output": {
        "console_log": False,
        "file_log": True,
        "save_detections": True,
        "save_interval": 60,  # Save less frequently
        "save_on_detection": True,  # Only save when objects detected
        "image_quality": 75,  # Lower quality to save space
    }
}

# ==================== COCO CLASSES ====================
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ==================== COLOR SCHEMES ====================
COLOR_SCHEMES = {
    "vibrant": {
        "person": (255, 0, 0),
        "cell phone": (0, 255, 0),
        "laptop": (255, 255, 0),
        "cat": (255, 0, 255),
        "dog": (0, 255, 255),
        "car": (255, 165, 0),
        "bottle": (128, 0, 128),
        "chair": (0, 128, 128),
        "default": (200, 200, 200)
    }
}

# ==================== STORAGE MANAGER ====================
class StorageManager:
    """Manages image storage with automatic cleanup"""
    
    def __init__(self, folder_path, max_size_mb):
        self.folder_path = Path(folder_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.folder_path.mkdir(exist_ok=True)
        self.lock = Lock()
    
    def get_folder_size(self):
        """Calculate total size of folder in bytes"""
        total = 0
        for file in self.folder_path.glob("detection_*.jpg"):
            try:
                total += file.stat().st_size
            except:
                pass
        return total
    
    def cleanup_old_files(self):
        """Remove oldest files if storage exceeds limit"""
        with self.lock:
            current_size = self.get_folder_size()
            
            if current_size <= self.max_size_bytes:
                return
            
            files = sorted(
                self.folder_path.glob("detection_*.jpg"),
                key=lambda x: x.stat().st_mtime
            )
            
            for file in files:
                if current_size <= self.max_size_bytes * 0.8:
                    break
                try:
                    file_size = file.stat().st_size
                    file.unlink()
                    current_size -= file_size
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
    
    def save_image(self, image, prefix="detection"):
        """Save image with automatic cleanup"""
        self.cleanup_old_files()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = self.folder_path / f"{prefix}_{timestamp}.jpg"
        
        with self.lock:
            cv2.imwrite(str(filename), image, 
                       [cv2.IMWRITE_JPEG_QUALITY, CONFIG["output"]["image_quality"]])
        
        return filename

# ==================== THERMAL MANAGER ====================
class ThermalManager:
    """Monitor and manage Pi temperature"""
    
    def __init__(self, threshold=70):
        self.threshold = threshold
        self.is_raspberry = self._is_raspberry_pi()
        self.throttled = False
    
    def _is_raspberry_pi(self):
        """Check if running on Raspberry Pi"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                return 'raspberry' in f.read().lower()
        except:
            return False
    
    def get_temperature(self):
        """Get CPU temperature"""
        if not self.is_raspberry:
            return None
        
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                return temp
        except:
            return None
    
    def should_throttle(self):
        """Check if system should throttle due to temperature"""
        temp = self.get_temperature()
        if temp is None:
            return False
        
        if temp > self.threshold:
            self.throttled = True
            return True
        elif temp < self.threshold - 5:
            self.throttled = False
            return False
        
        return self.throttled

# ==================== MODEL DOWNLOADER ====================
class ModelDownloader:
    """Download TensorFlow Lite models"""
    
    @staticmethod
    def download_model():
        """Download MobileNet SSD TFLite model and labels"""
        import urllib.request
        
        model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
        
        print("Downloading MobileNet SSD model (4MB)...")
        
        try:
            # Download model
            import zipfile
            import io
            
            response = urllib.request.urlopen(model_url)
            zip_data = response.read()
            
            # Extract
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
                zip_ref.extractall(".")
            
            # Rename files
            if os.path.exists("detect.tflite"):
                print("✅ Model downloaded: detect.tflite")
            
            if os.path.exists("labelmap.txt"):
                print("✅ Labels downloaded: labelmap.txt")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            print("\nManual download instructions:")
            print("1. Download: https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip")
            print("2. Extract detect.tflite and labelmap.txt to current directory")
            return False

# ==================== DETECTION SYSTEM ====================
class DetectionSystem:
    def __init__(self, config):
        self.config = config
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.cap = None
        self.detection_history = {}
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.platform = self.detect_platform()
        self.fps = 0
        self.processing_times = deque(maxlen=30)
        self.labels = []
        
        # Initialize managers
        self.storage_manager = StorageManager(
            self.config["system"]["data_folder"],
            self.config["system"]["max_storage_mb"]
        )
        
        self.thermal_manager = ThermalManager(
            self.config["system"]["temp_threshold"]
        ) if self.config["system"]["thermal_throttle"] else None
        
        # Initialize system
        self.setup_logging()
        self.init_system()

    def detect_platform(self):
        """Detect the current platform"""
        import platform
        system = platform.system().lower()
        
        if system == "linux":
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = Path(self.config["system"]["data_folder"]) / f"detection_log_{timestamp}.txt"
            self.log_file.parent.mkdir(exist_ok=True)
            
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
        
        if self.config["system"]["log_to_file"] and self.config["output"]["file_log"]:
            with open(self.log_file, 'a') as f:
                f.write(formatted_msg + "\n")
    
    def load_labels(self):
        """Load COCO labels"""
        label_file = self.config['detection']['labels']
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            # Remove index prefix if present (e.g., "0 person" -> "person")
            self.labels = [label.split(' ', 1)[-1] if ' ' in label else label for label in self.labels]
            
            self.log_message(f"Loaded {len(self.labels)} labels")
        else:
            self.log_message("Label file not found, using default COCO classes", "WARNING")
            self.labels = COCO_CLASSES
    
    def init_system(self):
        """Initialize the detection system"""
        self.log_message(f"Initializing on platform: {self.platform}")
        
        # Check if model exists
        model_path = self.config['detection']['model']
        if not os.path.exists(model_path):
            self.log_message(f"Model not found: {model_path}", "WARNING")
            self.log_message("Attempting to download model...")
            
            if not ModelDownloader.download_model():
                self.log_message("Failed to download model", "ERROR")
                sys.exit(1)
        
        # Load labels
        self.load_labels()
        
        # Load TFLite model
        self.log_message(f"Loading TFLite model: {model_path}")
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape
            input_shape = self.input_details[0]['shape']
            self.log_message(f"Model input shape: {input_shape}")
            
            # Warm up model
            dummy_input = np.zeros(input_shape, dtype=np.uint8)
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            
            self.log_message("Model loaded and warmed up successfully")
            
        except Exception as e:
            self.log_message(f"Error loading model: {e}", "ERROR")
            sys.exit(1)
        
        # Initialize camera
        self.init_camera()
    
    def init_camera(self):
        """Initialize camera"""
        self.log_message(f"Initializing camera {self.config['camera']['device_id']}")
        
        if self.platform == "raspberry":
            backend = self.config['camera']['backend']
        else:
            backend = cv2.CAP_ANY
            self.log_message(f"Non-Pi platform detected ({self.platform}), using CAP_ANY backend")
        
        self.cap = cv2.VideoCapture(self.config['camera']['device_id'], backend)
        
        if not self.cap.isOpened():
            self.log_message("Error: Could not open camera", "ERROR")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        self.log_message(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.frame_fps:.1f} FPS")
    
    def get_color(self, label):
        """Get color for a label"""
        scheme = COLOR_SCHEMES.get(self.config['visualization']['color_scheme'], COLOR_SCHEMES['vibrant'])
        return scheme.get(label, scheme['default'])
    
    def detect_objects(self, frame):
        """Run object detection on frame"""
        # Resize frame to model input size
        input_size = self.config['performance']['input_size']
        input_frame = cv2.resize(frame, input_size)
        
        # Convert BGR to RGB
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        
        # Expand dimensions to match model input
        input_frame = np.expand_dims(input_frame, axis=0)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_frame)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensors
        # MobileNet SSD outputs: [boxes, classes, scores, num_detections]
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding box coordinates
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class indices
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence scores
        
        return boxes, classes, scores
    
    def process_detections(self, frame, boxes, classes, scores):
        """Process detection results"""
        current_detections = {}
        height, width = frame.shape[:2]
        
        for i in range(len(scores)):
            if scores[i] < self.config['detection']['confidence']:
                continue
            
            # Get class label
            class_id = int(classes[i])
            if class_id < len(self.labels):
                label = self.labels[class_id]
            else:
                label = f"class_{class_id}"
            
            # Get bounding box
            ymin, xmin, ymax, xmax = boxes[i]
            
            # Convert to pixel coordinates
            x1 = int(xmin * width)
            y1 = int(ymin * height)
            x2 = int(xmax * width)
            y2 = int(ymax * height)
            
            # Calculate center
            obj_cx = int((x1 + x2) / 2)
            obj_cy = int((y1 + y2) / 2)
            
            detection_data = {
                'label': label,
                'confidence': float(scores[i]),
                'center': (obj_cx, obj_cy),
                'bbox': (x1, y1, x2, y2),
                'offset': (obj_cx - self.center_x, obj_cy - self.center_y),
            }
            
            current_detections[f"{label}_{len(current_detections)}"] = detection_data
        
        return current_detections
    
    def draw_detections(self, frame, detections):
        """Draw all detections on frame"""
        for detection in detections.values():
            label = detection['label']
            conf = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            color = self.get_color(label)
            
            if self.config['visualization']['show_bbox']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{label} {conf:.2f}"
            cv2.putText(frame, label_text, 
                       (x1, y1 - 10 if y1 > 20 else y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_overlay(self, frame, detections):
        """Draw overlay information"""
        if self.config['visualization']['show_fps']:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        obj_count = len(detections)
        count_text = f"Objects: {obj_count}"
        cv2.putText(frame, count_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.thermal_manager:
            temp = self.thermal_manager.get_temperature()
            if temp:
                temp_color = (0, 255, 0) if temp < 60 else (0, 255, 255) if temp < 70 else (0, 0, 255)
                temp_text = f"Temp: {temp:.1f}C"
                cv2.putText(frame, temp_text, (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_color, 2)
    
    def save_detection_image(self, frame, detections):
        """Save detection image"""
        if not self.config['output']['save_detections']:
            return
        
        current_time = time.time()
        should_save = current_time - self.last_save_time >= self.config['output']['save_interval']
        
        if self.config['output']['save_on_detection'] and len(detections) == 0:
            should_save = False
        
        if should_save:
            try:
                filename = self.storage_manager.save_image(frame)
                self.log_message(f"Saved: {filename.name}")
                self.last_save_time = current_time
            except Exception as e:
                self.log_message(f"Error saving image: {e}", "ERROR")
    
    def run(self):
        """Main run loop"""
        self.log_message("Starting detection system...")
        self.log_message("Press 'q' to quit.")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        frame_skip_counter = 0
        render_counter = 0
        
        target_frame_time = 1.0 / self.config['performance']['throttle_fps'] if self.config['performance']['throttle_fps'] > 0 else 0

        try:
            while True:
                loop_start = time.time()
                
                # Check thermal throttling
                if self.thermal_manager and self.thermal_manager.should_throttle():
                    self.log_message("Thermal throttling active", "WARNING")
                    time.sleep(1.0)
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message("Error reading frame", "ERROR")
                    break
                
                self.frame_count += 1
                
                # Apply flips
                if self.config['camera']['flip_horizontal']:
                    frame = cv2.flip(frame, 1)
                if self.config['camera']['flip_vertical']:
                    frame = cv2.flip(frame, 0)
                
                # Frame skipping
                if frame_skip_counter > 0:
                    frame_skip_counter -= 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                frame_skip_counter = self.config['performance']['frame_skip']
                
                # Process frame
                process_start = time.time()
                
                boxes, classes, scores = self.detect_objects(frame)
                detections = self.process_detections(frame, boxes, classes, scores)
                self.processed_count += 1
                
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                
                # Render
                should_render = render_counter % self.config['visualization']['render_every_n'] == 0
                render_counter += 1
                
                if should_render:
                    self.draw_detections(frame, detections)
                    self.draw_overlay(frame, detections)
                    cv2.imshow("TFLite Detection System", frame)
                
                # Save detections
                self.save_detection_image(frame, detections)
                
                # Calculate FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    self.fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    
                    avg_process_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                    self.log_message(
                        f"FPS: {self.fps:.1f} | Proc: {avg_process_time*1000:.1f}ms | "
                        f"Detections: {len(detections)}"
                    )
                
                # Handle input
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Throttle FPS
                elapsed = time.time() - loop_start
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.log_message("Shutting down...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        runtime = time.time() - self.start_time
        self.log_message(f"System ran for {runtime:.1f} seconds")
        self.log_message(f"Captured: {self.frame_count} frames")
        self.log_message(f"Processed: {self.processed_count} frames")
        
        if self.processed_count > 0:
            avg_fps = self.processed_count / runtime
            self.log_message(f"Average FPS: {avg_fps:.1f}")

# ==================== MAIN EXECUTION ====================

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n[!] Ctrl+C detected. Shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("  TensorFlow Lite Detection System - Pi 3 Optimized")
    print("=" * 60)
    print(f"  Performance Mode: {CONFIG['performance']['mode']}")
    print(f"  Target FPS: {CONFIG['performance']['throttle_fps']}")
    print(f"  Model: {CONFIG['detection']['model']}")
    print(f"  Max Storage: {CONFIG['system']['max_storage_mb']} MB")
    print("=" * 60)
    print()
    
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
        print("\n" + "=" * 60)
        print("  Detection system terminated.")
        print("=" * 60)