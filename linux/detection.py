#!/usr/bin/env python3 
import cv2
import sys
import signal
import time
import os
import gc
from datetime import datetime
from pathlib import Path
from collections import deque
import threading
from threading import Thread, Lock, Event
from queue import Queue, Empty
import numpy as np
import urllib.request
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_loader import ConfigLoader

# ==================== LOAD CONFIGURATION ====================
CONFIG = ConfigLoader.load("config.json")

# Convert backend string to OpenCV constant
if CONFIG["camera"]["backend"] == "CAP_V4L2":
    CONFIG["camera"]["backend"] = cv2.CAP_V4L2
elif CONFIG["camera"]["backend"] == "CAP_ANY":
    CONFIG["camera"]["backend"] = cv2.CAP_ANY
else:
    CONFIG["camera"]["backend"] = cv2.CAP_ANY

# Import streaming server only if enabled
if CONFIG["system"].get("enable_streaming", False):
    try:
        from streaming_server import VideoStreamer
        STREAMING_AVAILABLE = True
    except ImportError:
        print("⚠️ Streaming server not available")
        STREAMING_AVAILABLE = False
else:
    STREAMING_AVAILABLE = False

# ==================== MEMORY MONITOR ====================
class MemoryMonitor:
    """Monitor system memory to prevent crashes"""
    def __init__(self, warning_threshold_mb=100):
        self.warning_threshold = warning_threshold_mb * 1024 * 1024
        self.last_check = time.time()
        
    def get_free_memory(self):
        """Get free memory in bytes"""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'MemAvailable:' in line:
                        return int(line.split()[1]) * 1024
        except:
            return None
        return None
    
    def should_cleanup(self):
        """Check if we need to run garbage collection"""
        current_time = time.time()
        if current_time - self.last_check < 5.0:  # Check every 5 seconds
            return False
        
        self.last_check = current_time
        free_mem = self.get_free_memory()
        
        if free_mem and free_mem < self.warning_threshold:
            print(f"⚠️ Low memory: {free_mem/1024/1024:.1f}MB free")
            return True
        return False

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

    def save_image(self, image, prefix="detection", detections=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        self.critical_threshold = threshold + 5
        self.last_warning = 0
        
    def get_temperature(self):
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read()) / 1000.0
        except:
            return None
    
    def check_temperature(self):
        temp = self.get_temperature()
        if temp is None:
            return "normal"
        
        current_time = time.time()
        
        if temp > self.critical_threshold:
            if current_time - self.last_warning > 10:  # Limit warnings
                print(f"🔥 CRITICAL: Temperature {temp:.1f}°C")
                self.last_warning = current_time
            return "critical"
        elif temp > self.threshold:
            if current_time - self.last_warning > 30:  # Limit warnings
                print(f"⚠️ Warning: Temperature {temp:.1f}°C")
                self.last_warning = current_time
            return "warning"
        return "normal"

# ==================== DETECTION SYSTEM ====================
class DetectionSystem:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.cap = None
        
        # Initialize managers
        self.storage_manager = StorageManager(
            self.config["system"]["data_folder"],
            self.config["system"]["max_storage_mb"]
        )
        
        self.thermal_manager = ThermalManager(
            self.config["system"]["temp_threshold"]
        )
        
        self.memory_monitor = MemoryMonitor()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.last_status_time = time.time()
        self.last_save_time = 0
        self.last_gc_time = time.time()
        
        # Streaming
        self.streamer = None
        if STREAMING_AVAILABLE and config["system"].get("enable_streaming", False):
            try:
                streaming_port = config["system"].get("streaming_port", 5000)
                max_clients = config["system"].get("max_streaming_clients", 3)
                self.streamer = VideoStreamer(port=streaming_port, max_clients=max_clients)
                print(f"🌐 Streaming enabled on port {streaming_port}")
            except Exception as e:
                print(f"⚠️ Failed to start streaming: {e}")
        
        self.setup_logging()
        self.init_system()

    def setup_logging(self):
        if self.config["system"]["log_to_file"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = Path(self.config["system"]["data_folder"]) / f"log_{timestamp}.txt"
            self.log_file.parent.mkdir(exist_ok=True)

    def init_system(self):
        print("\n🚀 Initializing Detection System...")
        
        # Load YOLOv8 model
        try:
            print("🧠 Loading YOLOv8 Nano model...")
            self.model = YOLO('yolov8n.pt')
            self.model.conf = self.config['detection']['confidence']
            self.model.iou = self.config['performance']['nms_threshold']
            print("✅ Model loaded")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            sys.exit(1)
        
        # Initialize camera
        self.init_camera()

    def init_camera(self):
        print("📷 Initializing camera...")
        try:
            self.cap = cv2.VideoCapture(self.config['camera']['device_id'], 
                                       self.config['camera']['backend'])
            
            if not self.cap.isOpened():
                print("❌ Could not open camera!")
                sys.exit(1)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try YUYV first (less CPU), fallback to MJPG
            fourcc_code = cv2.VideoWriter_fourcc(*self.config['camera'].get('fourcc', 'YUYV'))
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ Camera: {actual_width}x{actual_height} @ {self.config['camera']['fps']} FPS")
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            sys.exit(1)

    def detect_objects_simple(self, frame):
        """Simplified detection to reduce memory usage"""
        inference_size = self.config["performance"].get("inference_size", 256)
        
        # Resize for inference
        if frame.shape[1] != inference_size:
            inference_frame = cv2.resize(frame, (inference_size, inference_size))
        else:
            inference_frame = frame
        
        # Run inference with minimal settings
        results = self.model(inference_frame, 
                            verbose=False, 
                            imgsz=inference_size,
                            max_det=self.config['detection'].get('max_detections', 10))
        
        boxes = []
        classes = []
        scores = []
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < self.config['detection']['confidence']:
                    continue
                
                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Scale to original frame
                height, width = frame.shape[:2]
                scale_x = width / inference_size
                scale_y = height / inference_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                boxes.append([x1, y1, x2, y2])
                classes.append(int(box.cls[0]))
                scores.append(conf)
        
        return boxes, classes, scores

    def process_detections(self, boxes, classes, scores):
        """Process detection results"""
        detections = {}
        coco_classes = [
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
        
        for i in range(len(scores)):
            class_id = classes[i]
            label = coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"
            
            detections[f"{label}_{i}"] = {
                'label': label,
                'confidence': scores[i],
                'bbox': tuple(boxes[i]),
            }
        
        return detections

    def display_status(self, fps, temp, detections):
        """Display status with minimal overhead"""
        current_time = time.time()
        if current_time - self.last_status_time < self.config["output"].get("status_update_interval", 5):
            return
        
        self.last_status_time = current_time
        
        status = f"📊 {fps:4.1f}FPS | 🌡️{temp:4.1f}°C"
        
        if detections:
            det_counts = {}
            for det in detections.values():
                label = det['label']
                det_counts[label] = det_counts.get(label, 0) + 1
            
            if det_counts:
                main_label = next(iter(det_counts))
                count = det_counts[main_label]
                status += f" | 🎯 {count}x{main_label[:3]}"
                if len(det_counts) > 1:
                    status += f"(+{len(det_counts)-1})"
        
        print(status)

    def run(self):
        print("\n" + "="*60)
        print("🚀 PI 5 DETECTION SYSTEM (Optimized for Stability)")
        print("="*60)
        print("• Press Ctrl+C to stop")
        print(f"• Detection threshold: {self.config['detection']['confidence']}")
        print(f"• Target FPS: {self.config['performance']['throttle_fps']}")
        
        if self.streamer:
            print(f"• Streaming: ENABLED")
        print("="*60 + "\n")
        
        fps_frame_count = 0
        fps_start_time = time.time()
        frame_skip_counter = 0
        
        try:
            while True:
                loop_start = time.time()
                
                # Check temperature
                temp_status = self.thermal_manager.check_temperature()
                if temp_status == "critical":
                    print("🔥 Critical temperature! Cooling down...")
                    time.sleep(2.0)
                    continue
                elif temp_status == "warning":
                    # Skip frames when hot
                    if frame_skip_counter < 2:
                        frame_skip_counter += 1
                        time.sleep(0.5)
                        continue
                
                # Check memory
                if self.memory_monitor.should_cleanup():
                    gc.collect()
                    self.last_gc_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Apply frame skipping
                if frame_skip_counter > 0:
                    frame_skip_counter -= 1
                    continue
                frame_skip_counter = self.config['performance']['frame_skip']
                
                # Run detection
                boxes, classes, scores = self.detect_objects_simple(frame)
                detections = self.process_detections(boxes, classes, scores)
                
                # Update streaming
                if self.streamer:
                    temp = self.thermal_manager.get_temperature()
                    self.streamer.update_frame(
                        frame=frame,
                        detections=detections,
                        fps=self.fps,
                        frame_count=self.frame_count,
                        temperature=temp,
                        processing_time=0
                    )
                
                # Save image if needed
                current_time = time.time()
                if (self.config['output']['save_detections'] and 
                    current_time - self.last_save_time >= self.config['output']['save_interval'] and
                    (not self.config['output']['save_on_detection'] or len(detections) > 0)):
                    
                    self.storage_manager.save_image(frame, detections=detections)
                    self.last_save_time = current_time
                
                # Update FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 2.0:
                    self.fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                    
                    # Display status
                    temp = self.thermal_manager.get_temperature()
                    if temp:
                        self.display_status(self.fps, temp, detections)
                
                # Throttle to target FPS
                elapsed = time.time() - loop_start
                target_time = 1.0 / self.config['performance']['throttle_fps']
                sleep_time = target_time - elapsed
                if sleep_time > 0.001:
                    time.sleep(min(sleep_time, 0.1))  # Cap sleep time
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping...")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n🧹 Cleaning up...")
        if self.cap:
            self.cap.release()
        if self.streamer:
            self.streamer.stop()
        
        runtime = time.time() - self.start_time
        print(f"⏱️ Runtime: {runtime:.1f}s")
        print(f"🎬 Frames processed: {self.frame_count}")
        if self.frame_count > 0:
            print(f"📈 Avg FPS: {self.frame_count/runtime:.1f}")
        print("✅ Cleanup complete")

# ==================== MAIN ====================
def signal_handler(sig, frame):
    print("\n\n⏹️ Ctrl+C detected")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🔧 Checking system...")
    
    # Check power supply
    try:
        with open('/sys/class/power_supply/rpi_poe_power/current_now', 'r') as f:
            current = int(f.read()) / 1000  # mA
            if current < 2500:
                print(f"⚠️ Power supply may be weak: {current:.0f}mA")
    except:
        pass
    
    # Check temperature
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
            print(f"🌡️ Current temperature: {temp:.1f}°C")
            if temp > 70:
                print("⚠️ Temperature already high!")
    except:
        print("⚠️ Could not read temperature")
    
    try:
        system = DetectionSystem(CONFIG)
        system.run()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())