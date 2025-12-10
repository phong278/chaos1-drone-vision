#!/usr/bin/env python3
"""
Test streaming server locally - No Pi needed!
Simulates full Pi 4 detection system on your laptop.
"""

import cv2
import numpy as np
import time
import threading
from flask import Flask, Response
import socket

print("🧪 LOCAL STREAMING TEST - No Pi Required")
print("="*60)

# ==================== MOCK DETECTION SYSTEM ====================
class MockDetectionSystem:
    """Simulates Pi 4 detection without needing Pi hardware"""
    
    def __init__(self):
        self.frame_count = 0
        self.fps = 15.0
        self.temperature = 45.0
        self.detections = []
        
        # Create test patterns
        self.test_patterns = self._create_test_patterns()
        self.pattern_index = 0
        
        print("✅ Mock detection system created")
        print("   • Simulated camera feed")
        print("   • Mock object detection")
        print("   • Simulated temperature")
    
    def _create_test_patterns(self):
        """Create rotating test patterns"""
        patterns = []
        
        # Pattern 1: Person on left
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (200, 300), (0, 0, 255), -1)  # Red person
        cv2.putText(img, "PERSON", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        patterns.append(img)
        
        # Pattern 2: Car on right
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (400, 200), (550, 300), (255, 0, 0), -1)  # Blue car
        cv2.putText(img, "CAR", (400, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        patterns.append(img)
        
        # Pattern 3: Multiple objects
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (150, 150), (250, 350), (0, 0, 255), -1)  # Person
        cv2.rectangle(img, (350, 200), (500, 280), (255, 0, 0), -1)  # Car
        cv2.rectangle(img, (300, 350), (350, 450), (0, 255, 0), -1)  # Chair
        cv2.putText(img, "MULTIPLE OBJECTS", (200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        patterns.append(img)
        
        # Pattern 4: Empty scene
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "CLEAR - NO OBJECTS", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        patterns.append(img)
        
        return patterns
    
    def get_next_frame(self):
        """Get next test frame"""
        frame = self.test_patterns[self.pattern_index].copy()
        self.pattern_index = (self.pattern_index + 1) % len(self.test_patterns)
        self.frame_count += 1
        
        # Add frame counter
        cv2.putText(frame, f"Test Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add simulated FPS
        cv2.putText(frame, f"Simulated FPS: {self.fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add temperature
        cv2.putText(frame, f"Sim Temp: {self.temperature:.1f}°C", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def simulate_detection(self, frame):
        """Simulate object detection"""
        self.detections = []
        
        # Simulate detections based on pattern
        if self.pattern_index == 0:  # Person pattern
            self.detections = [{
                'label': 'person',
                'confidence': 0.85,
                'bbox': [100, 100, 200, 300]
            }]
        elif self.pattern_index == 1:  # Car pattern
            self.detections = [{
                'label': 'car',
                'confidence': 0.72,
                'bbox': [400, 200, 550, 300]
            }]
        elif self.pattern_index == 2:  # Multiple objects
            self.detections = [
                {'label': 'person', 'confidence': 0.85, 'bbox': [150, 150, 250, 350]},
                {'label': 'car', 'confidence': 0.72, 'bbox': [350, 200, 500, 280]},
                {'label': 'chair', 'confidence': 0.65, 'bbox': [300, 350, 350, 450]}
            ]
        # Pattern 3: No detections
        
        return self.detections

# ==================== STREAMING SERVER ====================
class LocalStreamingServer:
    """Full streaming server test - identical to Pi version"""
    
    def __init__(self, port=5000):
        self.port = port
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.detection_info = {
            'fps': 15.0,
            'frame_count': 0,
            'detections': [],
            'temperature': 45.0,
            'processing_time': 35.0
        }
        
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>🧪 LOCAL TEST - Pi 4 Simulation</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background: #1a1a1a;
                        color: white;
                    }
                    .container {
                        max-width: 800px;
                        margin: 0 auto;
                    }
                    .header {
                        text-align: center;
                        margin-bottom: 20px;
                        background: linear-gradient(45deg, #4a00e0, #8e2de2);
                        padding: 20px;
                        border-radius: 10px;
                    }
                    .warning {
                        background: #ff9900;
                        color: black;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 10px 0;
                        text-align: center;
                    }
                    .video-container {
                        background: black;
                        padding: 10px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                        border: 2px solid #4a00e0;
                    }
                    .stats, .detections {
                        background: #2a2a2a;
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 10px;
                    }
                    .detection-item {
                        margin: 5px 0;
                        padding: 5px;
                        background: #3a3a3a;
                        border-radius: 5px;
                    }
                    .person { border-left: 4px solid #ff0000; }
                    .car { border-left: 4px solid #0088ff; }
                    .chair { border-left: 4px solid #00ff00; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🧪 Pi 4 Drone System - LOCAL TEST</h1>
                        <p>Simulating Raspberry Pi 4 detection on your laptop</p>
                    </div>
                    
                    <div class="warning">
                        ⚠️ <strong>TEST MODE</strong> - This is a simulation running locally.<br>
                        No Pi hardware required. Testing streaming functionality only.
                    </div>
                    
                    <div class="video-container">
                        <img src="/video_feed" width="640" height="480" style="display: block; margin: 0 auto;">
                    </div>
                    
                    <div class="stats">
                        <h3>📊 Simulated System Stats</h3>
                        <div id="stats">Loading...</div>
                    </div>
                    
                    <div class="detections">
                        <h3>🎯 Simulated Detections</h3>
                        <div id="detections">No objects detected</div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 20px; color: #888;">
                        <p>Test Pattern Rotation: Person → Car → Multiple → Clear (loop)</p>
                    </div>
                </div>
                
                <script>
                    function updateStats() {
                        fetch('/stats')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('stats').innerHTML = `
                                    <p>🎥 FPS: ${data.fps.toFixed(1)} | Frame: ${data.frame_count}</p>
                                    <p>🌡️ Temperature: ${data.temperature.toFixed(1)}°C (simulated)</p>
                                    <p>⚡ Processing: ${data.processing_time.toFixed(1)}ms</p>
                                `;
                                
                                if (data.detections && data.detections.length > 0) {
                                    let html = '<p>Detected Objects:</p>';
                                    data.detections.forEach(det => {
                                        const cls = det.label.toLowerCase();
                                        html += `
                                            <div class="detection-item ${cls}">
                                                <strong>${det.label}</strong>: ${(det.confidence * 100).toFixed(1)}% confidence
                                                <br>
                                                <small>Bounding Box: [${det.bbox.join(', ')}]</small>
                                            </div>
                                        `;
                                    });
                                    document.getElementById('detections').innerHTML = html;
                                } else {
                                    document.getElementById('detections').innerHTML = 
                                        '<p>No objects detected (clear test pattern)</p>';
                                }
                            })
                            .catch(error => console.error('Error:', error));
                    }
                    
                    setInterval(updateStats, 1000);
                    updateStats();
                </script>
            </body>
            </html>
            '''
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats')
        def get_stats():
            return self.detection_info
    
    def update_frame(self, frame, detections, fps=15.0, frame_count=0, temperature=45.0, processing_time=35.0):
        """Update frame with simulated detections"""
        frame_with_overlay = frame.copy()
        
        # Draw detection boxes
        detection_list = []
        for det in detections:
            label = det['label']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            # Color coding
            colors = {
                'person': (0, 0, 255),    # Red
                'car': (255, 0, 0),       # Blue
                'chair': (0, 255, 0),     # Green
            }
            color = colors.get(label.lower(), (255, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame_with_overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label} {conf:.0%}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_with_overlay, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame_with_overlay, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            detection_list.append(det)
        
        # Add "SIMULATION" watermark
        cv2.putText(frame_with_overlay, "🧪 SIMULATION - NO PI REQUIRED", (150, 420),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Update
        with self.frame_lock:
            self.latest_frame = frame_with_overlay
            self.detection_info.update({
                'fps': fps,
                'frame_count': frame_count,
                'detections': detection_list,
                'temperature': temperature,
                'processing_time': processing_time
            })
    
    def generate_frames(self):
        """Generate frames for streaming"""
        while True:
            with self.frame_lock:
                if self.latest_frame is None:
                    # Placeholder
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Starting test stream...", 
                               (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                else:
                    ret, buffer = cv2.imencode('.jpg', self.latest_frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    
    def run(self):
        """Start the streaming server"""
        # Get local IP
        def get_local_ip():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 1))
                ip = s.getsockname()[0]
            except:
                ip = '127.0.0.1'
            finally:
                s.close()
            return ip
        
        ip = get_local_ip()
        
        # Start in background thread
        def start_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        print("\n" + "="*60)
        print("📡 LOCAL STREAMING SERVER STARTED")
        print("="*60)
        print(f"On THIS computer, open:")
        print(f"  • http://localhost:{self.port}")
        print(f"  • http://{ip}:{self.port}")
        print("="*60)
        print("Test Patterns:")
        print("  1. Person (red box)")
        print("  2. Car (blue box)")
        print("  3. Multiple objects")
        print("  4. Empty scene")
        print("="*60)
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        return server_thread

# ==================== MAIN TEST ====================
def run_local_test(duration_seconds=30):
    """Run complete local test"""
    print("🚀 Starting Pi 4 Streaming Simulation...")
    
    # Create mock system
    mock_system = MockDetectionSystem()
    
    # Start streaming server
    streamer = LocalStreamingServer(port=5000)
    streamer.run()
    
    # Give server time to start
    time.sleep(2)
    
    print("🎬 Starting test sequence...")
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            # Get next test frame
            frame = mock_system.get_next_frame()
            
            # Simulate detection
            detections = mock_system.simulate_detection(frame)
            
            # Update streamer
            streamer.update_frame(
                frame=frame,
                detections=detections,
                fps=mock_system.fps,
                frame_count=mock_system.frame_count,
                temperature=mock_system.temperature,
                processing_time=35.0
            )
            
            # Print progress
            if frame_count % 10 == 0:
                pattern_names = ["PERSON", "CAR", "MULTIPLE", "CLEAR"]
                current_pattern = pattern_names[(mock_system.pattern_index - 1) % 4]
                print(f"[{frame_count:4d}] Pattern: {current_pattern:10} | "
                      f"Detections: {len(detections)}")
            
            frame_count += 1
            time.sleep(0.067)  # ~15 FPS
            
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    
    print("\n✅ Local streaming test complete!")
    print(f"   Frames processed: {frame_count}")
    print(f"   Test duration: {time.time() - start_time:.1f}s")
    print("\n📋 Verification:")
    print("   1. Open http://localhost:5000 in browser")
    print("   2. Verify video stream appears")
    print("   3. Verify detection boxes show")
    print("   4. Verify stats update in real-time")
    print("   5. Test on phone/other device: http://YOUR_IP:5000")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=30, help='Test duration in seconds')
    parser.add_argument('--port', type=int, default=5000, help='Port for streaming server')
    args = parser.parse_args()
    
    run_local_test(args.duration)