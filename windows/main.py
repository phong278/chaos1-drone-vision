#!/usr/bin/env python3
"""
YOLOv4-tiny detection system, optimized for Raspberry Pi 4 (OpenCV DNN).
Modified to print detected objects to terminal and save pictures with storage management.
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
import zipfile
import io

# ==================== CONFIGURATION ====================
CONFIG = {
    "system": {
        "platform": "auto",
        "log_to_file": True,
        "data_folder": "detection_data",
        "max_storage_mb": 500,
        "thermal_throttle": True,
        "temp_threshold": 75,
    },
    "performance": {
        "mode": "balanced",
        "throttle_fps": 15,
        "frame_skip": 0,
        "resize_input": True,
        "input_size": (416, 416),
        "use_threading": True,
        "frame_buffer_size": 2,
        "nms_threshold": 0.45,
        "conf_threshold": 0.35,
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
        "model_cfg": "yolov4-tiny.cfg",
        "model_weights": "yolov4-tiny.weights",
        "labels": "coco.names",
        "confidence": 0.35,
        "max_classes": 80,
    },
    "visualization": {
        "show_fps": True,
        "show_bbox": True,
        "show_center": False,
        "color_scheme": "vibrant",
        "render_every_n": 1,
    },
    "output": {
        "console_log": True,  # CHANGED: Enable console logging
        "file_log": True,
        "save_detections": True,
        "save_interval": 2,   # CHANGED: Save every 2 seconds (adjust as needed)
        "save_on_detection": True,
        "image_quality": 80,
        "print_detections": True,  # NEW: Print detections to terminal
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

# ==================== COLOR SCHEMES ====================
COLOR_SCHEMES = {
    "vibrant": {
        "default": (200, 200, 200),
    }
}

# ==================== STORAGE MANAGER ====================
class StorageManager:
    """Manages image storage with automatic cleanup"""
    def __init__(self, folder_path, max_size_mb):
        self.folder_path = Path(folder_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.folder_path.mkdir(exist_ok=True, parents=True)
        self.lock = Lock()
        print(f"📁 Storage folder: {self.folder_path.absolute()}")
        print(f"💾 Max storage: {max_size_mb} MB")

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
            
            print(f"⚠️  Storage limit reached ({current_size/1024/1024:.1f}MB > {self.max_size_bytes/1024/1024:.1f}MB)")
            print("🧹 Cleaning up old files...")
            
            files = sorted(self.folder_path.glob("detection_*.jpg"),
                           key=lambda x: x.stat().st_mtime)
            deleted_count = 0
            for file in files:
                if current_size <= self.max_size_bytes * 0.8:  # Clean to 80%
                    break
                try:
                    file_size = file.stat().st_size
                    file.unlink()
                    current_size -= file_size
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Error deleting {file}: {e}")
            
            if deleted_count > 0:
                print(f"✅ Deleted {deleted_count} old files")
                print(f"📊 New folder size: {current_size/1024/1024:.1f}MB")

    def save_image(self, image, prefix="detection", detections=None):
        self.cleanup_old_files()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Add detection info to filename if available
        if detections and len(detections) > 0:
            labels = [d['label'] for d in detections.values()]
            label_str = "_".join(sorted(set(labels))[:3])  # First 3 unique labels
            filename = self.folder_path / f"{prefix}_{label_str}_{timestamp}.jpg"
        else:
            filename = self.folder_path / f"{prefix}_{timestamp}.jpg"
        
        with self.lock:
            cv2.imwrite(str(filename), image,
                       [cv2.IMWRITE_JPEG_QUALITY, CONFIG["output"]["image_quality"]])
        
        # Print save confirmation
        if detections and len(detections) > 0:
            print(f"💾 Saved: {filename.name} (Detections: {len(detections)})")
        else:
            print(f"💾 Saved: {filename.name}")
        
        return filename

# ==================== THERMAL MANAGER ====================
class ThermalManager:
    """Monitor and manage Pi temperature"""
    def __init__(self, threshold=75):
        self.threshold = threshold
        self.is_raspberry = self._is_raspberry_pi()
        self.throttled = False

    def _is_raspberry_pi(self):
        try:
            with open('/proc/device-tree/model', 'r') as f:
                return 'raspberry' in f.read().lower()
        except:
            return False

    def get_temperature(self):
        if not self.is_raspberry:
            return None
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
        if temp > self.threshold:
            self.throttled = True
            return True
        elif temp < self.threshold - 5:
            self.throttled = False
            return False
        return self.throttled

# ==================== MODEL DOWNLOADER ====================
class ModelDownloader:
    """Download YOLO tiny files if missing (best-effort)"""
    @staticmethod
    def download_file(url, dest):
        try:
            print(f"Downloading {url} -> {dest} ...")
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
        if not ok:
            print("\n⚠️  Could not download all model files automatically.")
        return ok

# ==================== CAMERA GRABBER (threaded) ====================
class CameraGrabber(Thread):
    """Threaded camera grabber to maximize capture throughput."""
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
        self.thermal_manager = ThermalManager(self.config["system"]["temp_threshold"]) \
            if self.config["system"]["thermal_throttle"] else None
        self.platform = self.detect_platform()
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.processing_times = deque(maxlen=30)
        self.last_print_time = time.time()
        self.print_interval = 1.0  # Print detections every 1 second
        self.setup_logging()
        self.init_system()

    def detect_platform(self):
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
        if self.config["system"]["log_to_file"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = Path(self.config["system"]["data_folder"]) / f"detection_log_{timestamp}.txt"
            self.log_file.parent.mkdir(exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write(f"Detection System Log - Started at {timestamp}\n")
                f.write(f"Platform: {self.platform}\n")
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
        """Print detected objects to terminal"""
        if not detections or not self.config["output"]["print_detections"]:
            return
        
        current_time = time.time()
        if current_time - self.last_print_time < self.print_interval:
            return
        
        self.last_print_time = current_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{'='*50}")
        print(f"📸 DETECTIONS @ {timestamp} (Frame {self.frame_count})")
        print(f"{'='*50}")
        
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
            print(f"  • {label.upper()}: {len(det_list)} instance(s), avg confidence: {avg_conf:.1%}")
            for det in det_list[:2]:  # Show first 2 instances
                bbox = det['bbox']
                print(f"    └─ {det['confidence']:.1%} at [{bbox[0]},{bbox[1]}]→[{bbox[2]},{bbox[3]}]")
        
        print(f"{'='*50}")

    def load_labels(self):
        label_file = self.config['detection']['labels']
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]
            self.log_message(f"Loaded {len(self.labels)} labels from {label_file}")
        else:
            self.log_message("Label file not found, using default COCO classes", "WARNING")
            self.labels = COCO_CLASSES

    def init_system(self):
        self.log_message(f"Initializing on platform: {self.platform}")
        cfg = self.config['detection']['model_cfg']
        weights = self.config['detection']['model_weights']
        names = self.config['detection']['labels']

        if not (os.path.exists(cfg) and os.path.exists(weights)):
            self.log_message("Model files not found, attempting to download (best-effort).", "WARNING")
            ModelDownloader.ensure_models(cfg, weights, names)

        if not os.path.exists(cfg) or not os.path.exists(weights):
            self.log_message("Model files missing. Exiting.", "ERROR")
            print("Please ensure YOLO cfg/weights are present or allow the downloader to fetch them.")
            sys.exit(1)

        self.load_labels()

        try:
            self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            except Exception:
                pass
            ln = self.net.getLayerNames()
            layer_indices = self.net.getUnconnectedOutLayers()

            # Handle different return types on different platforms
            if layer_indices.ndim == 2:
                # Linux/Raspberry Pi returns 2D array
                self.output_layer_names = [ln[i[0] - 1] for i in layer_indices]
            else:
                # Windows returns 1D array
                self.output_layer_names = [ln[i - 1] for i in layer_indices]
            self.log_message("YOLO network loaded successfully.")
        except Exception as e:
            self.log_message(f"Failed to load YOLO network: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        self.init_camera()

        self.frame_queue = Queue(maxsize=self.config["performance"]["frame_buffer_size"])
        self.stop_event = ThreadEvent()
        if self.config["performance"]["use_threading"]:
            self.grabber = CameraGrabber(self.cap, self.frame_queue, self.stop_event)
            self.grabber.start()

    def init_camera(self):
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
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        self.log_message(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.cap.get(cv2.CAP_PROP_FPS):.1f} FPS")

    def get_color(self, label):
        scheme = COLOR_SCHEMES.get(self.config['visualization']['color_scheme'], COLOR_SCHEMES['vibrant'])
        return scheme.get(label, scheme['default'])

    def detect_objects(self, frame):
        net = self.net
        input_w, input_h = self.config["performance"]["input_size"]
        conf_thresh = self.config["performance"]["conf_threshold"]
        nms_thresh = self.config["performance"]["nms_threshold"]

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_w, input_h), swapRB=True, crop=False)
        net.setInput(blob)
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
        current_detections = {}
        height, width = frame.shape[:2]

        for i in range(len(scores)):
            if scores[i] < self.config['detection']['confidence']:
                continue

            class_id = int(classes[i])
            label = f"class_{class_id}"
            if class_id < len(self.labels):
                label = self.labels[class_id]
            elif 0 <= class_id - 1 < len(self.labels):
                label = self.labels[class_id - 1]

            ymin, xmin, ymax, xmax = boxes[i]

            x1 = int(max(0, xmin) * width)
            y1 = int(max(0, ymin) * height)
            x2 = int(min(1, xmax) * width)
            y2 = int(min(1, ymax) * height)

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
        if not self.config['output']['save_detections']:
            return
        
        current_time = time.time()
        should_save = current_time - getattr(self, "last_save_time", 0) >= self.config['output']['save_interval']
        
        if self.config['output']['save_on_detection'] and len(detections) == 0:
            should_save = False
            
        if should_save:
            try:
                # Pass detections to include in filename
                filename = self.storage_manager.save_image(frame, detections=detections)
                self.last_save_time = current_time
            except Exception as e:
                self.log_message(f"Error saving image: {e}", "ERROR")

    def run(self):
        self.log_message("Starting YOLO detection system...")
        self.log_message("Press 'q' to quit.")
        fps_start_time = time.time()
        fps_frame_count = 0
        frame_skip_counter = 0
        render_counter = 0
        target_frame_time = 1.0 / self.config['performance']['throttle_fps'] if self.config['performance']['throttle_fps'] > 0 else 0

        try:
            while True:
                loop_start = time.time()

                if self.thermal_manager and self.thermal_manager.should_throttle():
                    self.log_message("Thermal throttling active", "WARNING")
                    time.sleep(1.0)
                    continue

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

                if self.config['camera']['flip_horizontal']:
                    frame = cv2.flip(frame, 1)
                if self.config['camera']['flip_vertical']:
                    frame = cv2.flip(frame, 0)

                if frame_skip_counter > 0:
                    frame_skip_counter -= 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                frame_skip_counter = self.config['performance']['frame_skip']

                process_start = time.time()
                boxes, classes, scores = self.detect_objects(frame)
                detections = self.process_detections(frame, boxes, classes, scores)
                self.processed_count += 1
                process_time = time.time() - process_start
                self.processing_times.append(process_time)

                # PRINT DETECTIONS TO TERMINAL
                self.print_detections_to_terminal(detections)

                should_render = render_counter % self.config['visualization']['render_every_n'] == 0
                render_counter += 1
                if should_render:
                    self.draw_detections(frame, detections)
                    self.draw_overlay(frame, detections)
                    cv2.imshow("YOLO Detection (OpenCV DNN)", frame)

                # SAVE IMAGE WITH STORAGE MANAGEMENT
                self.save_detection_image(frame, detections)

                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    elapsed = time.time() - fps_start_time
                    self.fps = fps_frame_count / elapsed if elapsed > 0 else 0.0
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    avg_process_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                    self.log_message(
                        f"FPS: {self.fps:.1f} | Proc: {avg_process_time*1000:.1f}ms | "
                        f"Detections: {len(detections)}"
                    )

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                elapsed = time.time() - loop_start
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self.cleanup()

    def cleanup(self):
        self.log_message("Shutting down...")
        if getattr(self, "stop_event", None):
            self.stop_event.set()
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


# Simple Event wrapper for clean thread stop
class ThreadEvent:
    def __init__(self):
        self._flag = False

    def is_set(self):
        return self._flag

    def set(self):
        self._flag = True


# ==================== MAIN EXECUTION ====================
def signal_handler(sig, frame):
    print("\n\n[!] Ctrl+C detected. Shutting down...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print("=" * 60)
    print("  YOLO Detection System - Optimized for Pi4 (OpenCV DNN)")
    print("=" * 60)
    print(f"  Performance Mode: {CONFIG['performance']['mode']}")
    print(f"  Target FPS: {CONFIG['performance']['throttle_fps']}")
    print(f"  Model cfg: {CONFIG['detection']['model_cfg']}")
    print(f"  Model weights: {CONFIG['detection']['model_weights']}")
    print(f"  Save images: {'YES' if CONFIG['output']['save_detections'] else 'NO'}")
    print(f"  Max storage: {CONFIG['system']['max_storage_mb']}MB")
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