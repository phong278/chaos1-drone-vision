# streaming_server.py
from flask import Flask, Response
import cv2
import threading
import time
import socket

app = Flask(__name__)

class VideoStreamer:
    def __init__(self, port=5000):
        self.port = port
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_streaming = False
        self.detection_info = {
            'fps': 0.0,
            'frame_count': 0,
            'detections': [],
            'temperature': None
        }
        
        # Setup routes
        @app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>🚁 Pi 4 Drone Camera</title>
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
                    }
                    .video-container {
                        background: black;
                        padding: 10px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }
                    .stats {
                        background: #2a2a2a;
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 10px;
                    }
                    .detections {
                        background: #2a2a2a;
                        padding: 15px;
                        border-radius: 10px;
                    }
                    .detection-item {
                        margin: 5px 0;
                        padding: 5px;
                        background: #3a3a3a;
                        border-radius: 5px;
                    }
                </style>
                <meta http-equiv="refresh" content="30">
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚁 Pi 4 Drone Camera Live Stream</h1>
                        <p>Real-time object detection from Raspberry Pi 4</p>
                    </div>
                    
                    <div class="video-container">
                        <img src="/video_feed" width="640" height="480">
                    </div>
                    
                    <div class="stats">
                        <h3>📊 System Stats</h3>
                        <div id="stats">Loading stats...</div>
                    </div>
                    
                    <div class="detections">
                        <h3>🎯 Current Detections</h3>
                        <div id="detections">No objects detected</div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 20px; color: #888;">
                        <p>Streaming from Raspberry Pi 4 | Auto-refreshes every 30 seconds</p>
                    </div>
                </div>
                
                <script>
                    function updateStats() {
                        fetch('/stats')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('stats').innerHTML = `
                                    <p>🎥 FPS: ${data.fps.toFixed(1)} | Frame: ${data.frame_count}</p>
                                    <p>🌡️ Temperature: ${data.temperature || 'N/A'}°C</p>
                                    <p>📈 Processing: ${data.processing_time || '0'}ms</p>
                                `;
                                
                                if (data.detections && data.detections.length > 0) {
                                    let html = '';
                                    data.detections.forEach(det => {
                                        html += `
                                            <div class="detection-item">
                                                <strong>${det.label}</strong>: ${(det.confidence * 100).toFixed(1)}% confidence
                                                <br>
                                                <small>Location: [${det.bbox[0]}, ${det.bbox[1]}] → [${det.bbox[2]}, ${det.bbox[3]}]</small>
                                            </div>
                                        `;
                                    });
                                    document.getElementById('detections').innerHTML = html;
                                } else {
                                    document.getElementById('detections').innerHTML = '<p>No objects detected</p>';
                                }
                            })
                            .catch(error => console.error('Error:', error));
                    }
                    
                    // Update stats every 2 seconds
                    setInterval(updateStats, 2000);
                    updateStats(); // Initial call
                </script>
            </body>
            </html>
            '''
        
        @app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/stats')
        def get_stats():
            return {
                'fps': self.detection_info['fps'],
                'frame_count': self.detection_info['frame_count'],
                'detections': self.detection_info['detections'],
                'temperature': self.detection_info['temperature'],
                'processing_time': self.detection_info.get('processing_time', 0)
            }
    
    def update_frame(self, frame, detections, fps=0.0, frame_count=0, temperature=None, processing_time=0):
        """Update the latest frame with detections"""
        frame_with_overlay = frame.copy()
        
        # Draw detections
        detection_list = []
        for det in detections.values():
            label = det['label']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            # Choose color based on label
            if 'person' in label.lower():
                color = (0, 0, 255)  # Red
            elif 'car' in label.lower():
                color = (255, 0, 0)  # Blue
            elif 'chair' in label.lower():
                color = (0, 255, 0)  # Green
            else:
                color = (255, 255, 0)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(frame_with_overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label} {conf:.0%}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_with_overlay, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            cv2.putText(frame_with_overlay, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add to detection list for JSON
            detection_list.append({
                'label': label,
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
        
        # Add info overlay
        overlay_text = f"FPS: {fps:.1f} | Frame: {frame_count}"
        cv2.putText(frame_with_overlay, overlay_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if temperature:
            temp_text = f"Temp: {temperature:.1f}°C"
            cv2.putText(frame_with_overlay, temp_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if temperature < 70 else (0, 255, 255) if temperature < 80 else (0, 0, 255), 
                       2)
        
        # Update latest frame
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
        """Generate video frames for streaming"""
        while True:
            with self.frame_lock:
                if self.latest_frame is None:
                    # Send a placeholder frame
                    placeholder = self._create_placeholder()
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                else:
                    ret, buffer = cv2.imencode('.jpg', self.latest_frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)
    
    def _create_placeholder(self):
        """Create a placeholder frame when no camera feed"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Waiting for camera feed...", (100, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "Pi 4 Drone Detection System", (80, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return img
    
    def run(self):
        """Start the streaming server"""
        self.is_streaming = True
        
        # Get IP address
        ip = self._get_ip_address()
        
        # Start server in background thread
        def run_server():
            app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        print("\n" + "="*60)
        print("📡 STREAMING SERVER STARTED")
        print("="*60)
        print(f"Local:    http://localhost:{self.port}")
        print(f"Network:  http://{ip}:{self.port}")
        print(f"On your laptop, open: http://{ip}:{self.port}")
        print("="*60)
        print("Note: Make sure Pi and laptop are on same WiFi network!")
        print("="*60 + "\n")
        
        return server_thread
    
    def _get_ip_address(self):
        """Get Pi's IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))
            ip = s.getsockname()[0]
        except:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

# For standalone testing
if __name__ == "__main__":
    import numpy as np
    
    print("Testing streaming server...")
    streamer = VideoStreamer(port=5000)
    streamer.run()
    
    # Test with dummy frames
    import time
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "Test Stream", (200, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    while True:
        streamer.update_frame(test_img, {}, fps=10.0, frame_count=123)
        time.sleep(0.1)