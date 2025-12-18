#!/usr/bin/env python3
import cv2
import sys
import signal
import time
import os
from datetime import datetime
from pathlib import Path
from collections import deque
from threading import Thread, Lock, Event
from queue import Queue, Empty
import numpy as np
import urllib.request
from ultralytics import YOLO  # Changed to Ultralytics YOLOv8

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from streaming_server import VideoStreamer
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
        
        try:
            stat = os.statvfs(self.folder_path)
            free_space = stat.f_bavail * stat.f_frsize
            free_mb = free_space / (1024 * 1024)
            print(f"💿 Free space: {free_mb:.0f} MB")
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
            
            print(f"⚠️ Storage limit reached ({current_size/1024/1024:.1f}MB)")
            print("🧹 Cleaning up old files...")
            
            files = sorted(self.folder_path.glob("detection_*.jpg"),
                           key=lambda x: x.stat().st_mtime)
            deleted_count = 0
            deleted_size = 0
            
            for file in files:
                if current_size <= self.max_size_bytes * 0.8:
                    break
                try:
                    file_size = file.stat().st_size
                    file.unlink()
                    current_size -= file_size
                    deleted_count += 1
                    deleted_size += file_size
                except Exception as e:
                    print(f"❌ Error deleting {file}: {e}")
            
            if deleted_count > 0:
                print(f"✅ Deleted {deleted_count} files ({deleted_size/1024/1024:.1f}MB)")

    def save_image(self, image, prefix="detection", detections=None):
        self.cleanup_old_files()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if detections and len(detections) > 0:
            labels = [d['label'][:3] for d in detections.values()]
            label_str = "_".join(sorted(set(labels))[:2])
            filename = self.folder_path / f"{prefix}_{label_str}_{timestamp}.jpg"
        else:
            filename = self.folder_path / f"{prefix}_{timestamp}.jpg"
        
        with self.lock:
            cv2.imwrite(str(filename), image,
                       [cv2.IMWRITE_JPEG_QUALITY, CONFIG["output"]["image_quality"]])
        
        return filename

# ==================== THERMAL MANAGER ====================
class ThermalManager:
    """Monitor and manage Pi temperature with dynamic throttling"""
    def __init__(self, threshold=75):
        self.threshold = threshold
        self.throttled = False
        self.temp_history = deque(maxlen=10)
        print(f"🌡️ Thermal threshold: {threshold}°C")

    def get_temperature(self):
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                self.temp_history.append(temp)
                return temp
        except:
            return None

    def get_average_temperature(self):
        if len(self.temp_history) == 0:
            return None
        return sum(self.temp_history) / len(self.temp_history)

    def should_throttle(self):
        temp = self.get_temperature()
        if temp is None:
            return False
        
        avg_temp = self.get_average_temperature()
        
        # Dynamic throttling based on trend
        if avg_temp and len(self.temp_history) > 5:
            temp_trend = temp - list(self.temp_history)[0]
            
            if temp > self.threshold - 5 and temp_trend > 1.0:
                # Rapidly heating
                if not self.throttled:
                    print(f"📈 Heating rapidly: {temp:.1f}°C (trend: +{temp_trend:.1f}°C)")
                self.throttled = True
                return True
        
        if temp > self.threshold:
            if not self.throttled:
                print(f"🔥 Thermal throttling: {temp:.1f}°C")
            self.throttled = True
            return True
        elif temp < self.threshold - 8:
            if self.throttled:
                print(f"❄️ Throttling deactivated: {temp:.1f}°C")
            self.throttled = False
            return False
        
        return self.throttled

# ==================== MODEL DOWNLOADER ====================
class ModelDownloader:
    """Download YOLOv8 Nano model if missing"""
    @staticmethod
    def download_file(url, dest):
        try:
            print(f"⬇️ Downloading {dest.split('/')[-1]}...")
            urllib.request.urlretrieve(url, dest)
            print("✅ Done.")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    @staticmethod
    def ensure_yolov8_nano():
        model_path = Path("yolov8n.pt")
        if not model_path.exists():
            print("📦 YOLOv8 Nano model not found, downloading...")
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            return ModelDownloader.download_file(url, str(model_path))
        return True

# ==================== CAMERA GRABBER ====================
class CameraGrabber(Thread):
    """Threaded camera frame grabber with optimized buffer"""
    def __init__(self, cap, queue, stop_event, buffer_size=2):
        super().__init__(daemon=True)
        self.cap = cap
        self.queue = queue
        self.stop_event = stop_event
        self.buffer_size = buffer_size
        self.last_frame = None
        self.frame_lock = Lock()

    def run(self):
        # Clear camera buffer initially
        for _ in range(5):
            self.cap.read()
        
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)  # Reduced sleep
                continue
            
            with self.frame_lock:
                self.last_frame = frame
            
            try:
                if self.queue.qsize() < self.buffer_size:
                    self.queue.put_nowait(frame)
                else:
                    # Drop oldest frame if queue is full
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(frame)
                    except:
                        pass
            except:
                pass

    def get_latest_frame(self):
        with self.frame_lock:
            return self.last_frame

# ==================== DETECTION SYSTEM ====================
class DetectionSystem:
    def __init__(self, config):
        self.config = config
        self.model = None  # Changed to YOLOv8 model
        self.cap = None
        self.frame_queue = None
        self.grabber = None
        self.stop_event = None
        
        # Initialize managers
        self.storage_manager = StorageManager(
            self.config["system"]["data_folder"],
            self.config["system"]["max_storage_mb"]
        )
        
        self.thermal_manager = ThermalManager(
            self.config["system"]["temp_threshold"]
        )
        
        # Performance tracking
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.processing_times = deque(maxlen=30)
        self.last_print_time = time.time()
        self.print_interval = 2.0
        self.last_save_time = 0
        
        # Frame cache for detection
        self.last_detection_frame = None
        self.last_detection_time = 0
        self.detection_cache_valid = False
        
        self.setup_logging()
        self.init_system()

        # Streaming setup
        self.streaming_enabled = config["system"].get("enable_streaming", False)
        self.streamer = None
        
        if self.streaming_enabled:
            print("\n🌐 Initializing streaming server...")
            streaming_port = config["system"].get("streaming_port", 5000)
            max_clients = config["system"].get("max_streaming_clients", 5)
            
            self.streamer = VideoStreamer(port=streaming_port, max_clients=max_clients)
            self.streamer.run()
            time.sleep(0.5)
            
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 1))
                ip = s.getsockname()[0]
                s.close()
                print(f"✅ Stream URL: http://{ip}:{streaming_port}")
                print(f"👥 Max viewers: {max_clients}")
            except:
                print(f"✅ Stream URL: http://YOUR_PI_IP:{streaming_port}")

    def setup_logging(self):
        if self.config["system"]["log_to_file"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = Path(self.config["system"]["data_folder"]) / f"log_{timestamp}.txt"
            self.log_file.parent.mkdir(exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write(f"Pi Detection Log - {timestamp}\n")
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
        """Print detected objects with optimization"""
        if not detections or not self.config["output"]["print_detections"]:
            return
        
        current_time = time.time()
        if current_time - self.last_print_time < self.print_interval:
            return
        
        self.last_print_time = current_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Simplified output for performance
        detection_summary = {}
        for det in detections.values():
            label = det['label']
            detection_summary[label] = detection_summary.get(label, 0) + 1
        
        if detection_summary:
            summary_str = ", ".join([f"{count}x {label}" for label, count in detection_summary.items()])
            print(f"[{timestamp}] 🎯 Detected: {summary_str}")

    def init_system(self):
        print("\n🚀 Initializing Detection System for Pi 5...")
        
        # Ensure YOLOv8 Nano model exists
        if not ModelDownloader.ensure_yolov8_nano():
            print("❌ Failed to get YOLOv8 Nano model!")
            sys.exit(1)

        try:
            print("🧠 Loading YOLOv8 Nano model...")
            self.model = YOLO('yolov8n.pt')
            
            # Configure model for inference
            self.model.conf = self.config['detection']['confidence']  # confidence threshold
            self.model.iou = self.config['performance']['nms_threshold']  # NMS IoU threshold
            
            print("✅ YOLOv8 Nano model loaded")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        self.init_camera()

        self.frame_queue = Queue(maxsize=self.config["performance"]["frame_buffer_size"])
        self.stop_event = Event()
        
        if self.config["performance"]["use_threading"]:
            self.grabber = CameraGrabber(self.cap, self.frame_queue, self.stop_event, 
                                         buffer_size=self.config["performance"]["frame_buffer_size"])
            self.grabber.start()
            print("🎥 Threaded frame capture enabled")

    def init_camera(self):
        print("📷 Initializing camera with optimized settings...")
        try:
            backend = self.config['camera']['backend']
            
            # Use V4L2 with optimized settings for Pi 5
            self.cap = cv2.VideoCapture(self.config['camera']['device_id'], backend)
            
            if not self.cap.isOpened():
                print(f"⚠️ Backend {backend} failed, trying default...")
                self.cap = cv2.VideoCapture(self.config['camera']['device_id'])
                if not self.cap.isOpened():
                    print("❌ Could not open camera!")
                    sys.exit(1)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
            
            # Pi 5 specific optimizations
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ Camera: {self.frame_width}x{self.frame_height} @ {self.config['camera']['fps']} FPS")
            print(f"📊 Camera backend: {self.cap.getBackendName()}")
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            sys.exit(1)

    def detect_objects(self, frame):
        """YOLOv8 detection with caching"""
        current_time = time.time()
        
        # Check if we can reuse cached detections
        if (self.detection_cache_valid and 
            self.last_detection_frame is not None and
            current_time - self.last_detection_time < 0.5):  # Cache valid for 500ms
            return self.last_detection_frame
        
        # Perform detection
        start_time = time.time()
        
        # Resize frame for faster inference if configured
        inference_size = self.config["performance"].get("inference_size", 320)
        if inference_size != self.frame_width:
            inference_frame = cv2.resize(frame, (inference_size, int(inference_size * self.frame_height / self.frame_width)))
        else:
            inference_frame = frame
        
        # Run YOLOv8 inference
        results = self.model(inference_frame, verbose=False, imgsz=inference_size)
        
        boxes = []
        classes = []
        scores = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box coordinates (relative)
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Normalize coordinates
                    h, w = frame.shape[:2]
                    x1_norm = x1 / w
                    y1_norm = y1 / h
                    x2_norm = x2 / w
                    y2_norm = y2 / h
                    
                    boxes.append([y1_norm, x1_norm, y2_norm, x2_norm])
                    classes.append(int(box.cls[0]))
                    scores.append(float(box.conf[0]))
        
        # Cache results
        self.last_detection_time = current_time
        self.last_detection_frame = (np.array(boxes, dtype=float), 
                                     np.array(classes, dtype=int), 
                                     np.array(scores, dtype=float))
        self.detection_cache_valid = True
        
        return self.last_detection_frame

    def process_detections(self, frame, boxes, classes, scores):
        """Process detection results with COCO class names"""
        current_detections = {}
        height, width = frame.shape[:2]

        for i in range(len(scores)):
            if scores[i] < self.config['detection']['confidence']:
                continue

            class_id = int(classes[i])
            
            # Use COCO class names
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
            
            label = coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"

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
        should_save = current_time - self.last_save_time >= self.config['output']['save_interval']
        
        if self.config['output']['save_on_detection'] and len(detections) == 0:
            should_save = False
            
        if should_save:
            try:
                filename = self.storage_manager.save_image(frame, detections=detections)
                self.log_message(f"Saved: {filename.name}")
                self.last_save_time = current_time
            except Exception as e:
                self.log_message(f"Error saving: {e}", "ERROR")

    def run(self):
        print("\n" + "="*70)
        print("🚀 PI 5 DETECTION SYSTEM STARTING (YOLOv8 Nano)")
        print("="*70)
        print("• Press Ctrl+C to stop")
        print(f"• Detection threshold: {self.config['detection']['confidence']}")
        print(f"• Target FPS: {self.config['performance']['throttle_fps']}")
        print(f"• Inference size: {self.config['performance'].get('inference_size', 320)}px")

        if self.streaming_enabled:
            print(f"• Streaming: ENABLED on port {self.config['system']['streaming_port']}")
        
        print("="*70 + "\n")

        fps_start_time = time.time()
        fps_frame_count = 0
        frame_skip_counter = 0
        last_stream_update = 0
        stream_update_interval = 0.033
        dynamic_frame_skip = 0

        try:
            while True:
                loop_start = time.time()

                # Thermal check with dynamic throttling
                if self.thermal_manager.should_throttle():
                    thermal_temp = self.thermal_manager.get_temperature()
                    avg_temp = self.thermal_manager.get_average_temperature()
                    
                    if avg_temp and avg_temp > self.config["system"]["temp_threshold"] - 2:
                        dynamic_frame_skip = min(dynamic_frame_skip + 1, 5)
                        if self.streaming_enabled and self.streamer:
                            stream_update_interval = 0.2  # Slow updates when hot
                        time.sleep(0.5)
                    else:
                        dynamic_frame_skip = max(dynamic_frame_skip - 1, 0)
                    continue
                else:
                    dynamic_frame_skip = max(dynamic_frame_skip - 1, 0)
                    stream_update_interval = 0.033

                # Get frame
                frame = None
                if self.config["performance"]["use_threading"]:
                    try:
                        frame = self.frame_queue.get(timeout=0.5)
                    except Empty:
                        # Try to get latest frame from grabber
                        if self.grabber:
                            frame = self.grabber.get_latest_frame()
                        if frame is None:
                            continue
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.log_message("Frame read error", "ERROR")
                        break

                self.frame_count += 1

                # Dynamic frame skipping based on load
                total_frame_skip = self.config['performance']['frame_skip'] + dynamic_frame_skip
                if frame_skip_counter > 0:
                    frame_skip_counter -= 1
                    continue
                frame_skip_counter = total_frame_skip

                # Detection with timing
                process_start = time.time()
                boxes, classes, scores = self.detect_objects(frame)
                detections = self.process_detections(frame, boxes, classes, scores)
                self.processed_count += 1
                process_time = time.time() - process_start
                self.processing_times.append(process_time)

                self.print_detections_to_terminal(detections)

                # Update stream with optimized intervals
                current_time = time.time()
                if self.streaming_enabled and self.streamer and \
                   (current_time - last_stream_update >= stream_update_interval):
                    
                    temp = self.thermal_manager.get_temperature()
                    self.streamer.update_frame(
                        frame=frame,
                        detections=detections,
                        fps=self.fps,
                        frame_count=self.frame_count,
                        temperature=temp,
                        processing_time=process_time * 1000
                    )
                    last_stream_update = current_time
                
                self.save_detection_image(frame, detections)

                # FPS calculation
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:  # Update every second
                    elapsed = time.time() - fps_start_time
                    self.fps = fps_frame_count / elapsed if elapsed > 0 else 0.0
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    
                    # Display status every 3 seconds
                    if int(time.time()) % 3 == 0:
                        temp = self.thermal_manager.get_temperature()
                        if temp:
                            client_info = ""
                            if self.streaming_enabled and self.streamer:
                                client_info = f" | 👥 {self.streamer.active_clients}"
                            
                            status_msg = f"📊 {self.fps:5.1f}FPS | 🌡️{temp:4.1f}°C | F{self.frame_count}{client_info}"
                            
                            if detections:
                                det_types = {}
                                for det in detections.values():
                                    label = det['label']
                                    det_types[label] = det_types.get(label, 0) + 1
                                
                                if det_types:
                                    main_det = next(iter(det_types))
                                    count = det_types[main_det]
                                    status_msg += f" | 🎯 {count}x{main_det[:3]}"
                                    if len(det_types) > 1:
                                        status_msg += f"(+{len(det_types)-1})"
                            
                            print(status_msg)

                # Adaptive sleep for target FPS
                elapsed = time.time() - loop_start
                target_time = 1.0 / self.config['performance']['throttle_fps']
                sleep_time = target_time - elapsed - 0.001  # Small buffer
                if sleep_time > 0.001:  # Only sleep if significant time left
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping...")
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        print("\n" + "="*50)
        print("🧹 Cleaning up...")
        if self.stop_event:
            self.stop_event.set()
        if self.cap:
            self.cap.release()
        if self.streamer:
            self.streamer.stop()
        
        runtime = time.time() - self.start_time
        print(f"⏱️ Runtime: {runtime:.1f}s")
        print(f"🎬 Total frames: {self.frame_count}")
        print(f"🔍 Processed frames: {self.processed_count}")
        if self.processed_count > 0:
            avg_fps = self.processed_count / runtime
            print(f"📈 Avg processing FPS: {avg_fps:.1f}")
            
            if self.processing_times:
                avg_process_time = sum(self.processing_times) / len(self.processing_times)
                print(f"⚡ Avg inference: {avg_process_time*1000:.1f}ms")
        print("="*50)

# ==================== MAIN ====================
def signal_handler(sig, frame):
    print("\n\n⏹️ Ctrl+C detected - Shutting down gracefully...")
    sys.exit(0)

def check_pi5_optimizations():
    """Check and suggest Pi 5 optimizations"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            
        if 'raspberry pi 5' in model:
            print("✅ Raspberry Pi 5 detected")
            print("💡 Pi 5 Optimizations enabled:")
            print("   • YOLOv8 Nano for better performance")
            print("   • Dynamic thermal throttling")
            print("   • Frame caching for efficiency")
            return True
        elif 'raspberry pi 4' in model:
            print("⚠️ Raspberry Pi 4 detected - Some optimizations may be less effective")
            return True
        else:
            print("⚠️ Unknown/Non-Pi hardware detected")
            return False
    except:
        print("⚠️ Could not verify hardware")
        return True

def main():
    """Main function with proper setup and teardown"""
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check hardware and warn
    if not check_pi5_optimizations():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Check for Ultralytics installation
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics package not found!")
        print("   Install with: pip install ultralytics")
        sys.exit(1)
    
    try:
        print("\n" + "="*60)
        print("Starting Pi 5 Object Detection System")
        print("="*60)
        
        system = DetectionSystem(CONFIG)
        system.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n👋 Program finished successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)