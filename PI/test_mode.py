#!/usr/bin/env python3
"""
TEST MODE for YOLO detection - Works on Windows/Mac/Linux
Simulates Pi 4 behavior for testing
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

# ==================== TEST CONFIGURATION ====================
CONFIG = {
    "system": {
        "platform": "test",
        "log_to_file": True,
        "data_folder": "test_detection_data",
        "max_storage_mb": 50,  # Smaller for testing
        "thermal_throttle": False,  # Disable for testing
        "temp_threshold": 75,
    },
    "performance": {
        "mode": "test",
        "throttle_fps": 5,          # Lower for testing
        "frame_skip": 0,
        "resize_input": True,
        "input_size": (320, 320),
        "use_threading": False,     # Disable threading for easier debugging
        "frame_buffer_size": 1,
        "nms_threshold": 0.45,
        "conf_threshold": 0.40,
    },
    "camera": {
        "device_id": 0,            # Your webcam
        "width": 640,
        "height": 480,
        "fps": 15,
        "flip_horizontal": False,
        "flip_vertical": False,
        "backend": cv2.CAP_ANY,    # Auto-detect
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
        "save_interval": 10,       # Save every 10 seconds
        "save_on_detection": True,
        "image_quality": 70,
        "print_detections": True,
        "show_gui": True,          # NEW: Show GUI for testing!
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
# ==================== MOCK THERMAL MANAGER ====================
class MockThermalManager:
    """Mock temperature monitoring for testing"""
    def __init__(self, threshold=75):
        self.threshold = threshold
    
    def get_temperature(self):
        # Return fake temperature (always cool for testing)
        return 45.0
    
    def should_throttle(self):
        return False  # Never throttle in test mode

# ==================== MOCK CAMERA (USING IMAGES/VIDEOS) ====================
class MockCamera:
    """Provides test frames without real camera"""
    
    @staticmethod
    def get_test_image():
        """Create a test image with known objects"""
        # Create a colored test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some colored rectangles that look like objects
        # Red rectangle (like a person)
        cv2.rectangle(img, (100, 100), (200, 300), (0, 0, 255), -1)
        
        # Blue rectangle (like a car)
        cv2.rectangle(img, (400, 200), (550, 300), (255, 0, 0), -1)
        
        # Green rectangle (like a chair)
        cv2.rectangle(img, (300, 350), (350, 450), (0, 255, 0), -1)
        
        # Add text labels
        cv2.putText(img, "Test Person", (100, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "Test Car", (400, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img, "Test Chair", (300, 330), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return img
    
    @staticmethod
    def load_test_video(video_path="test_video.mp4"):
        """Load a test video file"""
        if os.path.exists(video_path):
            return cv2.VideoCapture(video_path)
        else:
            print(f"Test video {video_path} not found, using static image")
            return None

# ==================== MODIFIED DETECTION SYSTEM ====================
class TestDetectionSystem:
    def __init__(self, config):
        self.config = config
        self.net = None
        self.labels = []
        self.cap = None
        self.test_mode = True
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.processing_times = deque(maxlen=30)
        self.last_print_time = time.time()
        self.print_interval = 2.0
        
        print("="*60)
        print("TEST MODE ACTIVATED")
        print("Simulating Pi 4 behavior on your current system")
        print("="*60)
        
        self.setup_logging()
        self.init_system()
    
    def setup_logging(self):
        if self.config["system"]["log_to_file"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = Path(self.config["system"]["data_folder"]) / f"test_log_{timestamp}.txt"
            self.log_file.parent.mkdir(exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write(f"Test Log - Started at {timestamp}\n")
                f.write("=" * 50 + "\n")
    
    def init_system(self):
        print("Loading YOLO model for testing...")
        
        # Download models if needed
        cfg = self.config['detection']['model_cfg']
        weights = self.config['detection']['model_weights']
        
        if not os.path.exists(cfg) or not os.path.exists(weights):
            print("Downloading model files...")
            # Use your ModelDownloader class here
            
        # Load labels
        self.labels = COCO_CLASSES  # Use fallback for testing
        
        try:
            # Load network
            self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layers (Windows/Linux compatible)
            ln = self.net.getLayerNames()
            layer_indices = self.net.getUnconnectedOutLayers()
            
            # Handle different array types
            if hasattr(layer_indices, 'ndim') and layer_indices.ndim == 2:
                self.output_layer_names = [ln[i[0] - 1] for i in layer_indices]
            else:
                self.output_layer_names = [ln[i - 1] for i in layer_indices]
            
            print(f"✅ Model loaded: {len(self.output_layer_names)} output layers")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Using mock detection for testing...")
            self.net = None
        
        # Initialize camera or test source
        self.init_camera()
    
    def init_camera(self):
        print("Initializing camera source...")
        
        # Try webcam first
        self.cap = cv2.VideoCapture(self.config['camera']['device_id'])
        
        if not self.cap.isOpened():
            print("Webcam not available, using test images...")
            self.cap = None
            self.test_frames = []
            
            # Generate some test frames
            for i in range(10):
                frame = MockCamera.get_test_image()
                # Add some variation
                cv2.putText(frame, f"Frame {i}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.test_frames.append(frame)
            
            self.frame_index = 0
        else:
            print("✅ Webcam connected")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
    
    def get_next_frame(self):
        """Get next frame from camera or test source"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera error, using test frames")
                self.cap = None
                return self.get_next_frame()
            return frame
        else:
            # Use test frames
            frame = self.test_frames[self.frame_index]
            self.frame_index = (self.frame_index + 1) % len(self.test_frames)
            return frame
    
    def detect_objects(self, frame):
        """Run YOLO detection or mock detection"""
        if self.net is None:
            # Mock detection for testing without model
            return self.mock_detection(frame)
        
        # Real YOLO detection
        input_w, input_h = self.config["performance"]["input_size"]
        conf_thresh = self.config["performance"]["conf_threshold"]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_w, input_h), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layer_names)
        
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
        
        # Apply NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.45)
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
    
    def mock_detection(self, frame):
        """Generate mock detections for testing"""
        height, width = frame.shape[:2]
        
        # Create some fake detections
        boxes = []
        classes = []
        scores = []
        
        # Fake person detection
        boxes.append([0.2, 0.2, 0.4, 0.6])  # ymin, xmin, ymax, xmax
        classes.append(0)  # person class ID
        scores.append(0.85)
        
        # Fake car detection
        boxes.append([0.4, 0.6, 0.7, 0.9])
        classes.append(2)  # car class ID
        scores.append(0.72)
        
        # Fake chair detection (sometimes)
        if self.frame_count % 3 == 0:
            boxes.append([0.7, 0.3, 0.9, 0.5])
            classes.append(56)  # chair class ID
            scores.append(0.65)
        
        return np.array(boxes, dtype=float), np.array(classes, dtype=int), np.array(scores, dtype=float)
    
    def process_detections(self, frame, boxes, classes, scores):
        """Process detection results"""
        current_detections = {}
        
        for i in range(len(scores)):
            if scores[i] < self.config['detection']['confidence']:
                continue
            
            class_id = int(classes[i])
            label = f"class_{class_id}"
            if class_id < len(self.labels):
                label = self.labels[class_id]
            
            ymin, xmin, ymax, xmax = boxes[i]
            height, width = frame.shape[:2]
            
            x1 = int(max(0, xmin) * width)
            y1 = int(max(0, ymin) * height)
            x2 = int(min(1, xmax) * width)
            y2 = int(min(1, ymax) * height)
            
            detection_data = {
                'label': label,
                'confidence': float(scores[i]),
                'bbox': (x1, y1, x2, y2),
            }
            
            current_detections[f"{label}_{i}"] = detection_data
        
        return current_detections
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame for display"""
        for detection in detections.values():
            label = detection['label']
            conf = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            # Choose color based on label
            if 'person' in label.lower():
                color = (0, 0, 255)  # Red
            elif 'car' in label.lower():
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 255, 0)  # Green
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add FPS counter
        if self.fps > 0:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run_test(self, duration_seconds=30):
        """Run test for specified duration"""
        print(f"\nStarting test run for {duration_seconds} seconds...")
        print("Press 'q' to quit early, 's' to save current frame")
        print("-" * 50)
        
        fps_start_time = time.time()
        fps_frame_count = 0
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            start_time = time.time()
            
            # Get frame
            frame = self.get_next_frame()
            self.frame_count += 1
            
            # Run detection
            boxes, classes, scores = self.detect_objects(frame)
            detections = self.process_detections(frame, boxes, classes, scores)
            self.processed_count += 1
            
            # Calculate processing time
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            
            # Print detections every 2 seconds
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                self.last_print_time = current_time
                print(f"\n[Frame {self.frame_count}] Detections:")
                if detections:
                    for label, det in detections.items():
                        print(f"  • {det['label']}: {det['confidence']:.1%}")
                else:
                    print("  No detections")
            
            # Display with detections drawn
            if self.config["output"]["show_gui"]:
                display_frame = self.draw_detections(frame.copy(), detections)
                cv2.imshow("Test Mode - Object Detection", display_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser requested quit")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"test_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")
            
            # FPS calculation
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                self.fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
        
        # Cleanup
        if self.config["output"]["show_gui"]:
            cv2.destroyAllWindows()
        
        if self.cap is not None:
            self.cap.release()
        
        # Print summary
        print("\n" + "="*50)
        print("TEST COMPLETE - SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.processed_count}")
        print(f"Total runtime: {time.time() - self.start_time:.1f}s")
        if self.processed_count > 0:
            avg_fps = self.processed_count / (time.time() - self.start_time)
            print(f"Average FPS: {avg_fps:.1f}")
            if self.processing_times:
                avg_process = sum(self.processing_times) / len(self.processing_times)
                print(f"Average process time: {avg_process*1000:.1f}ms")
        print("="*50)

# ==================== TEST SUITE ====================
def run_unit_tests():
    """Run basic unit tests"""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Check OpenCV import
    try:
        print("Test 1: OpenCV import... ", end="")
        import cv2
        print(f"✅ Version: {cv2.__version__}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
        tests_failed += 1
    
    # Test 2: Check NumPy import
    try:
        print("Test 2: NumPy import... ", end="")
        import numpy as np
        print(f"✅ Version: {np.__version__}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
        tests_failed += 1
    
    # Test 3: Create test image
    try:
        print("Test 3: Create test image... ", end="")
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (10, 10), (90, 90), (255, 0, 0), 2)
        print("✅ Created 100x100 test image")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
        tests_failed += 1
    
    # Test 4: Check model files
    try:
        print("Test 4: Check for model files... ", end="")
        required_files = ['yolov4-tiny.cfg', 'yolov4-tiny.weights', 'coco.names']
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print(f"⚠️ Missing: {missing}")
        else:
            print("✅ All model files found")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
        tests_failed += 1
    
    print(f"\n📊 Results: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0

def performance_test():
    """Test detection performance"""
    print("\n" + "="*60)
    print("PERFORMANCE TEST")
    print("="*60)
    
    # Create a simple test
    test_img = MockCamera.get_test_image()
    
    # Test detection speed
    import time
    times = []
    
    for i in range(5):
        start = time.time()
        # Create a dummy detection (simulate)
        height, width = test_img.shape[:2]
        boxes = np.array([[0.2, 0.2, 0.4, 0.6]])
        classes = np.array([0])
        scores = np.array([0.85])
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.1f}ms")
    
    avg_time = sum(times) / len(times)
    print(f"\n⏱️  Average: {avg_time*1000:.1f}ms per detection")
    print(f"📈 Theoretical FPS: {1/avg_time:.1f}")
    
    return avg_time

# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test YOLO detection system')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--perf', action='store_true', help='Run performance test only')
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    parser.add_argument('--nogui', action='store_true', help='Run without GUI display')
    
    args = parser.parse_args()
    
    # Set GUI based on argument
    CONFIG["output"]["show_gui"] = not args.nogui
    
    print("🧪 YOLO Detection System - TEST MODE")
    print("This simulates Pi 4 behavior on your current system")
    
    if args.unit:
        # Run unit tests only
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    
    if args.perf:
        # Run performance test only
        performance_test()
        sys.exit(0)
    
    # Run full test
    try:
        system = TestDetectionSystem(CONFIG)
        system.run_test(duration_seconds=args.duration)
        
        print("\n✅ Test completed successfully!")
        print("\nNext steps:")
        print("1. Verify detection works with your webcam")
        print("2. Test with different objects (person, car, etc.)")
        print("3. Adjust confidence thresholds in CONFIG")
        print("4. Test storage cleanup by filling detection_data folder")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)