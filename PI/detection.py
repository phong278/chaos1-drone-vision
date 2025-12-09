#!/usr/bin/env python3
"""
YOLOv4-tiny detection system - EXCLUSIVELY FOR RASPBERRY PI 4
Headless version with no GUI display - optimized for drone use
"""

import cv2
import sys
import signal
import time
import os
from datetime import datetime
from pathlib import Path
from collections import deque
from threading import Thread, Lock
from queue import Queue, Empty
import numpy as np
import urllib.request

# ==================== CONFIGURATION FOR PI 4 ====================
CONFIG = {
    "system": {
        "platform": "raspberry",
        "log_to_file": True,
        "data_folder": "detection_data",
        "max_storage_mb": 500,
        "thermal_throttle": True,
        "temp_threshold": 75,
    },
    "performance": {
        "mode": "balanced",
        "throttle_fps": 10,        # Reduced for Pi 4 stability
        "frame_skip": 1,           # Skip every other frame
        "resize_input": True,
        "input_size": (320, 320),  # Smaller for Pi 4 speed
        "use_threading": True,
        "frame_buffer_size": 2,
        "nms_threshold": 0.45,
        "conf_threshold": 0.40,    # Higher threshold for fewer false positives
    },
    "camera": {
        "device_id": 0,
        "width": 640,
        "height": 480,
        "fps": 15,                 # Lower FPS for Pi 4
        "flip_horizontal": False,
        "flip_vertical": False,
        "backend": cv2.CAP_V4L2,
        "buffer_size": 1,
    },
    "detection": {
        "model_cfg": "yolov4-tiny.cfg",
        "model_weights": "yolov4-tiny.weights",
        "labels": "coco.names",
        "confidence": 0.40,
        "max_classes": 80,
    },
    "output": {
        "console_log": True,
        "file_log": True,
        "save_detections": True,
        "save_interval": 5,        # Save every 5 seconds to reduce I/O
        "save_on_detection": True,
        "image_quality": 70,       # Lower quality for smaller files
        "print_detections": True,
    }
}

# ==================== COCO CLASSES (fallback) ====================
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
    'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# ==================== STORAGE MANAGER ====================
class StorageManager:
    """Manages image storage with automatic cleanup - Pi optimized"""
    def __init__(self, folder_path, max_size_mb):
        self.folder_path = Path(folder_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.folder_path.mkdir(exist_ok=True, parents=True)
        self.lock = Lock()
        print(f"Storage folder: {self.folder_path.absolute()}")
        print(f"Max storage: {max_size_mb} MB")
        
        # Check available disk space on Pi
        try:
            stat = os.statvfs(self.folder_path)
            free_space = stat.f_bavail * stat.f_frsize
            free_mb = free_space / (1024 * 1024)
            print(f"Free space: {free_mb:.0f} MB")
        except:
            pass

    def get_folder_size(self):
        total = 0
        for file in self.folder_path.glob("detection_*.jpg"):
            try:
                total += file.stat().st_size
            except:
                pass
        return total

    def cleanup_old_files(self):
        with self.lock:
            current_size = self.get_folder_size()
            if current_size <= self.max_size_bytes:
                return
            
            print(f"Storage limit reached ({current_size/1024/1024:.1f}MB > {self.max_size_bytes/1024/1024:.1f}MB)")
            print("Cleaning up old files...")
            
            files = sorted(self.folder_path.glob("detection_*.jpg"),
                           key=lambda x: x.stat().st_mtime)
            deleted_count = 0
            deleted_size = 0
            
            for file in files:
                if current_size <= self.max_size_bytes * 0.8:  # Clean to 80%
                    break
                try:
                    file_size = file.stat().st_size
                    file.unlink()
                    current_size -= file_size
                    deleted_count += 1
                    deleted_size += file_size
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            
            if deleted_count > 0:
                print(f"Deleted {deleted_count} files ({deleted_size/1024/1024:.1f}MB)")
                print(f"New folder size: {current_size/1024/1024:.1f}MB")

    def save_image(self, image, prefix="detection", detections=None):
        self.cleanup_old_files()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Simple filename for Pi (shorter, less CPU)
        if detections and len(detections) > 0:
            labels = [d['label'][:3] for d in detections.values()]  # First 3 chars
            label_str = "_".join(sorted(set(labels))[:2])  # Max 2 labels
            filename = self.folder_path / f"{prefix}_{label_str}_{timestamp}.jpg"
        else:
            filename = self.folder_path / f"{prefix}_{timestamp}.jpg"
        
        with self.lock:
            cv2.imwrite(str(filename), image,
                       [cv2.IMWRITE_JPEG_QUALITY, CONFIG["output"]["image_quality"]])
        
        return filename

# ==================== THERMAL MANAGER ====================
class ThermalManager:
    """Monitor and manage Pi temperature - Pi specific"""
    def __init__(self, threshold=75):
        self.threshold = threshold
        self.throttled = False

    def get_temperature(self):
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                return temp
        except:
            return None

    def should_throttle(self):
        temp = self.get_temperature()
        if temp is None:
            return False
        
        # Print warning if getting hot
        if temp > 70 and temp <= self.threshold:
            print(f"Warning: Temperature at {temp:.1f}°C")
        
        if temp > self.threshold:
            if not self.throttled:
                print(f"Thermal throttling activated: {temp:.1f}°C > {self.threshold}°C")
            self.throttled = True
            return True
        elif temp < self.threshold - 5:
            if self.throttled:
                print(f"Thermal throttling deactivated: {temp:.1f}°C")
            self.throttled = False
            return False
        
        return self.throttled

# ==================== MODEL DOWNLOADER ====================
class ModelDownloader:
    """Download YOLO tiny files if missing"""
    @staticmethod
    def download_file(url, dest):
        try:
            print(f"Downloading {dest.split('/')[-1]}...")
            urllib.request.urlretrieve(url, dest)
            print("Done.")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    @staticmethod
    def ensure_models(cfg_path, weights_path, names_path):
        ok = True
        urls = {
            "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
            "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            "names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        }
        
        if not os.path.exists(cfg_path):
            ok &= ModelDownloader.download_file(urls["cfg"], cfg_path)
        if not os.path.exists(weights_path):
            ok &= ModelDownloader.download_file(urls["weights"], weights_path)
        if not os.path.exists(names_path):
            ok &= ModelDownloader.download_file(urls["names"], names_path)
        
        return ok

# ==================== CAMERA GRABBER (threaded) ====================
class CameraGrabber(Thread):
    """Threaded camera grabber for Pi 4"""
    def __init__(self, cap, queue, stop_event):
        super().__init__(daemon=True)
        self.cap = cap
        self.queue = queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                self.queue.put_nowait(frame)
            except:
                # Queue full, drop frame (normal on Pi)
                pass

# ==================== DETECTION SYSTEM ====================
class DetectionSystem:
    def __init__(self, config):
        self.config = config
        self.net = None
        self.labels = []
        self.cap = None
        self.frame_queue = None
        self.grabber = None
        self.stop_event = None
        self.storage_manager = StorageManager(self.config["system"]["data_folder"],
                                              self.config["system"]["max_storage_mb"])
        self.thermal_manager = ThermalManager(self.config["system"]["temp_threshold"])
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.processing_times = deque(maxlen=30)
        self.last_print_time = time.time()
        self.print_interval = 2.0  # Print every 2 seconds to reduce spam
        self.setup_logging()
        self.init_system()

    def setup_logging(self):
        if self.config["system"]["log_to_file"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = Path(self.config["system"]["data_folder"]) / f"detection_log_{timestamp}.txt"
            self.log_file.parent.mkdir(exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write(f"Pi 4 Detection Log - Started at {timestamp}\n")
                f.write(f"Config: {self.config}\n")
                f.write("=" * 50 + "\n")

    def log_message(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        if self.config["output"]["console_log"]:
            print(formatted_msg)
        if self.config["system"]["log_to_file"] and self.config["output"]["file_log"]:
            with open(self.log_file, 'a') as f:
                f.write(formatted_msg + "\n")

    def print_detections_to_terminal(self, detections):
        """Print detected objects to terminal - Pi optimized"""
        if not detections or not self.config["output"]["print_detections"]:
            return
        
        current_time = time.time()
        if current_time - self.last_print_time < self.print_interval:
            return
        
        self.last_print_time = current_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Simple output for Pi
        print(f"\n┌{'─'*40}┐")
        print(f"│ DETECTIONS @ {timestamp} (Frame {self.frame_count})")
        print(f"├{'─'*40}┤")
        
        # Group detections by label
        detection_groups = {}
        for det in detections.values():
            label = det['label']
            if label not in detection_groups:
                detection_groups[label] = []
            detection_groups[label].append(det)
        
        # Print grouped detections
        for label, det_list in detection_groups.items():
            confidences = [d['confidence'] for d in det_list]
            avg_conf = sum(confidences) / len(confidences)
            print(f"│ • {label}: {len(det_list)}x ({avg_conf:.0%})")
        
        if not detection_groups:
            print(f"│ No objects detected")
        
        print(f"└{'─'*40}┘")

    def load_labels(self):
        label_file = self.config['detection']['labels']
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]
            self.log_message(f"Loaded {len(self.labels)} labels")
        else:
            self.log_message("Label file not found, using default COCO classes", "WARNING")
            self.labels = COCO_CLASSES

    def init_system(self):
        print("Initializing Pi 4 Detection System...")
        
        cfg = self.config['detection']['model_cfg']
        weights = self.config['detection']['model_weights']
        names = self.config['detection']['labels']

        if not (os.path.exists(cfg) and os.path.exists(weights)):
            print("Model files not found, downloading...")
            ModelDownloader.ensure_models(cfg, weights, names)

        if not os.path.exists(cfg) or not os.path.exists(weights):
            print("Model files missing. Exiting.")
            sys.exit(1)

        self.load_labels()

        try:
            print("Loading YOLO model...")
            self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
            
            # Pi 4 specific settings
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layers - Pi specific (returns 2D array)
            ln = self.net.getLayerNames()
            layer_indices = self.net.getUnconnectedOutLayers()
            self.output_layer_names = [ln[i[0] - 1] for i in layer_indices]
            
            print(f"Model loaded: {len(self.output_layer_names)} output layers")
            
        except Exception as e:
            print(f"Failed to load YOLO network: {e}")
            sys.exit(1)

        self.init_camera()

        # Setup threading for Pi 4
        self.frame_queue = Queue(maxsize=self.config["performance"]["frame_buffer_size"])
        self.stop_event = ThreadEvent()
        if self.config["performance"]["use_threading"]:
            self.grabber = CameraGrabber(self.cap, self.frame_queue, self.stop_event)
            self.grabber.start()

    def init_camera(self):
        print("Initializing camera...")
        try:
            # Force V4L2 for Pi
            self.cap = cv2.VideoCapture(self.config['camera']['device_id'], cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                print("Could not open camera")
                # Try default backend as fallback
                self.cap = cv2.VideoCapture(self.config['camera']['device_id'])
                if not self.cap.isOpened():
                    sys.exit(1)
            
            # Set camera properties for Pi
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.center_x = self.frame_width // 2
            self.center_y = self.frame_height // 2
            
            print(f"Camera: {self.frame_width}x{self.frame_height} @ {self.config['camera']['fps']} FPS")
            
        except Exception as e:
            print(f"Camera error: {e}")
            sys.exit(1)

    def detect_objects(self, frame):
        """YOLO detection optimized for Pi 4"""
        net = self.net
        input_w, input_h = self.config["performance"]["input_size"]
        conf_thresh = self.config["performance"]["conf_threshold"]
        nms_thresh = self.config["performance"]["nms_threshold"]

        # Create blob - smaller for Pi 4
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_w, input_h), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Run detection
        outs = net.forward(self.output_layer_names)

        height, width = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                if scores.size == 0:
                    continue
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf > conf_thresh:
                    # Convert to pixel coordinates
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(conf)
                    class_ids.append(class_id)

        if len(boxes) == 0:
            return np.zeros((0,4), dtype=float), np.array([], dtype=int), np.array([], dtype=float)

        # Apply NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
        if len(idxs) == 0:
            return np.zeros((0,4), dtype=float), np.array([], dtype=int), np.array([], dtype=float)

        out_boxes = []
        out_classes = []
        out_scores = []
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            xmin = max(0.0, float(x) / width)
            ymin = max(0.0, float(y) / height)
            xmax = min(1.0, float(x + w) / width)
            ymax = min(1.0, float(y + h) / height)
            out_boxes.append([ymin, xmin, ymax, xmax])
            out_classes.append(class_ids[i])
            out_scores.append(confidences[i])

        return np.array(out_boxes, dtype=float), np.array(out_classes, dtype=int), np.array(out_scores, dtype=float)

    def process_detections(self, frame, boxes, classes, scores):
        """Process detection results"""
        current_detections = {}
        height, width = frame.shape[:2]

        for i in range(len(scores)):
            if scores[i] < self.config['detection']['confidence']:
                continue

            class_id = int(classes[i])
            label = f"class_{class_id}"
            if class_id < len(self.labels):
                label = self.labels[class_id]

            ymin, xmin, ymax, xmax = boxes[i]

            x1 = int(max(0, xmin) * width)
            y1 = int(max(0, ymin) * height)
            x2 = int(min(1, xmax) * width)
            y2 = int(min(1, ymax) * height)

            detection_data = {
                'label': label,
                'confidence': float(scores[i]),
                'bbox': (x1, y1, x2, y2),
            }

            current_detections[f"{label}_{len(current_detections)}"] = detection_data

        return current_detections

    def save_detection_image(self, frame, detections):
        if not self.config['output']['save_detections']:
            return
        
        current_time = time.time()
        should_save = current_time - getattr(self, "last_save_time", 0) >= self.config['output']['save_interval']
        
        if self.config['output']['save_on_detection'] and len(detections) == 0:
            should_save = False
            
        if should_save:
            try:
                filename = self.storage_manager.save_image(frame, detections=detections)
                self.log_message(f"Saved: {filename.name}")
                self.last_save_time = current_time
            except Exception as e:
                self.log_message(f"Error saving image: {e}", "ERROR")

    def run(self):
        print("\n" + "="*50)
        print("PI 4 DETECTION SYSTEM STARTING")
        print("="*50)
        print("• Press Ctrl+C to stop")
        print("• Detections print every 2 seconds")
        print("• Images save every 5 seconds")
        print("="*50 + "\n")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        frame_skip_counter = 0

        try:
            while True:
                loop_start = time.time()

                # Thermal throttling check
                if self.thermal_manager.should_throttle():
                    print("Thermal throttling - slowing down...")
                    time.sleep(2.0)
                    continue

                # Get frame
                frame = None
                if self.config["performance"]["use_threading"] and self.frame_queue is not None:
                    try:
                        frame = self.frame_queue.get(timeout=1.0)
                    except Empty:
                        continue
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.log_message("Error reading frame", "ERROR")
                        break

                self.frame_count += 1

                # Skip frames if configured
                if frame_skip_counter > 0:
                    frame_skip_counter -= 1
                    continue
                frame_skip_counter = self.config['performance']['frame_skip']

                # Run detection
                process_start = time.time()
                boxes, classes, scores = self.detect_objects(frame)
                detections = self.process_detections(frame, boxes, classes, scores)
                self.processed_count += 1
                process_time = time.time() - process_start
                self.processing_times.append(process_time)

                # Print to terminal
                self.print_detections_to_terminal(detections)

                # Save image if needed
                self.save_detection_image(frame, detections)

                # FPS calculation
                fps_frame_count += 1
                if time.time() - fps_start_time >= 2.0:  # Update every 2 seconds
                    elapsed = time.time() - fps_start_time
                    self.fps = fps_frame_count / elapsed if elapsed > 0 else 0.0
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    
                    # Print status
                    temp = self.thermal_manager.get_temperature()
                    if temp:
                        print(f"Status: {self.fps:.1f} FPS | {temp:.1f}°C | Frame {self.frame_count}")

                # Small delay to prevent overheating
                elapsed = time.time() - loop_start
                target_time = 1.0 / self.config['performance']['throttle_fps']
                sleep_time = target_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nStopping detection system...")
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n" + "="*50)
        print("Cleaning up...")
        if self.stop_event:
            self.stop_event.set()
        if self.cap:
            self.cap.release()
        
        runtime = time.time() - self.start_time
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Frames processed: {self.processed_count}")
        if self.processed_count > 0:
            avg_fps = self.processed_count / runtime
            print(f"Average FPS: {avg_fps:.1f}")
        print("="*50)

class ThreadEvent:
    def __init__(self):
        self._flag = False

    def is_set(self):
        return self._flag

    def set(self):
        self._flag = True

# ==================== MAIN ====================
def signal_handler(sig, frame):
    print("\n\nCtrl+C detected. Shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if we're on Raspberry Pi
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'raspberry' not in model.lower():
                print("This script is for Raspberry Pi only!")
                print(f"   Detected: {model}")
                sys.exit(1)
            print(f"Running on: {model.strip()}")
    except:
        print("Could not verify Pi hardware (running anyway)")
    
    try:
        system = DetectionSystem(CONFIG)
        system.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()