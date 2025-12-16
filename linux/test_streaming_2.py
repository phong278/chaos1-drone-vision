#!/usr/bin/env python3
"""
Pi Camera Streaming Server
Uses ffplay to capture Raspberry Pi camera feed and stream it to webpage.
"""

import cv2
import numpy as np
import time
import threading
import queue
import json
import subprocess
from flask import Flask, Response, jsonify
import socket

print("📷 PI CAMERA STREAMING SERVER")
print("="*60)

# ==================== FFPLAY CAMERA CAPTURE ====================
class PiCameraCapture:
    """Capture Pi camera feed using ffplay"""
    
    def __init__(self, resolution=(640, 480), fps=25):
        self.width, self.height = resolution
        self.fps = fps
        self.frame_count = 0
        self.running = False
        self.capture_process = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Temperature tracking (for Pi)
        self.temperature = 45.0
        self.temp_update_time = time.time()
        
        print("✅ Pi Camera capture system created")
        print(f"   • Resolution: {self.width}x{self.height}")
        print(f"   • Target FPS: {self.fps}")
        print("   • Using ffplay for capture")
    
    def start_capture(self):
        """Start capturing from Pi camera using ffplay"""
        try:
            # FFPLAY command to capture from Pi camera
            # For Raspberry Pi camera module v2/v3
            ffplay_cmd = [
                'ffplay',
                '-f', 'v4l2',           # Video4Linux2 input
                '-video_size', f'{self.width}x{self.height}',
                '-framerate', str(self.fps),
                '-i', '/dev/video0',    # Pi camera device
                '-vf', 'format=bgr24',  # Output format OpenCV can read
                '-f', 'rawvideo',       # Raw video output
                '-loglevel', 'quiet',   # Quiet mode
                '-pipe:1'               # Pipe to stdout
            ]
            
            # Alternative for libcamera (Raspberry Pi OS Bullseye+)
            # ffplay_cmd = [
            #     'libcamera-vid',
            #     '--width', str(self.width),
            #     '--height', str(self.height),
            #     '--framerate', str(self.fps),
            #     '--codec', 'mjpeg',
            #     '-t', '0',  # Run indefinitely
            #     '-o', '-'   # Output to stdout
            # ]
            
            print(f"📹 Starting camera capture: {' '.join(ffplay_cmd)}")
            
            self.capture_process = subprocess.Popen(
                ffplay_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Large buffer for video
            )
            
            self.running = True
            
            # Start frame reading thread
            self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.read_thread.start()
            
            print("✅ Camera capture started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start camera capture: {e}")
            print("\n⚠️  If camera doesn't work, try:")
            print("   1. Check camera is connected: ls /dev/video*")
            print("   2. Install ffplay: sudo apt install ffmpeg")
            print("   3. Alternative: Use libcamera-vid instead")
            # Fall back to test pattern
            self._create_test_camera()
    
    def _create_test_camera(self):
        """Create a simulated camera feed if real camera fails"""
        print("⚠️  Creating simulated camera feed")
        self.running = True
        self.sim_thread = threading.Thread(target=self._simulate_camera, daemon=True)
        self.sim_thread.start()
    
    def _simulate_camera(self):
        """Simulate camera feed for testing"""
        print("🎬 Running simulated camera feed")
        while self.running:
            # Create dynamic test pattern
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Add timestamp
            current_time = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"PI CAMERA SIM - {current_time}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add moving elements
            t = time.time()
            
            # Moving circle
            x = int(self.width//2 + self.width//3 * np.sin(t))
            y = int(self.height//2 + self.height//3 * np.cos(t*0.7))
            cv2.circle(frame, (x, y), 40, (0, 100, 255), -1)
            
            # Moving rectangle
            x2 = int(self.width//4 + self.width//4 * np.sin(t*1.2))
            y2 = int(self.height//4 + self.height//4 * np.cos(t*0.9))
            cv2.rectangle(frame, (x2, y2), (x2+80, y2+80), (255, 100, 0), -1)
            
            # Update frame
            with self.frame_lock:
                self.latest_frame = frame
            
            time.sleep(1/self.fps)
    
    def _read_frames(self):
        """Read frames from ffplay process"""
        frame_size = self.width * self.height * 3  # BGR24
        
        print(f"📥 Reading frames (size: {frame_size} bytes)")
        
        while self.running and self.capture_process:
            try:
                # Read raw frame data
                raw_frame = self.capture_process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    if len(raw_frame) == 0:
                        # End of stream
                        print("⚠️  Camera stream ended")
                        break
                    # Skip incomplete frame
                    continue
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))
                
                with self.frame_lock:
                    self.latest_frame = frame
                
                self.frame_count += 1
                
                # Update temperature occasionally (simulated for now)
                if time.time() - self.temp_update_time > 5:
                    self.temperature = np.clip(40 + np.random.randn() * 5, 35, 75)
                    self.temp_update_time = time.time()
                
            except Exception as e:
                print(f"⚠️  Frame read error: {e}")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get the latest frame from camera"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        
        # Return black frame if no camera
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.capture_process:
            self.capture_process.terminate()
            self.capture_process.wait()
        print("📴 Camera capture stopped")

# ==================== STREAMING SERVER ====================
class CameraStreamingServer:
    """Streaming server for Pi camera feed"""
    
    def __init__(self, port=5000, max_clients=20):
        self.port = port
        self.max_clients = max_clients
        self.active_clients = 0
        self.client_lock = threading.Lock()
        
        # Camera setup
        self.camera = PiCameraCapture()
        self.camera.start_capture()
        
        # Frame management
        self.latest_frame = None
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()
        
        # Performance tracking
        self.stream_info = {
            'fps': 25.0,
            'frame_count': 0,
            'temperature': 45.0,
            'processing_time': 15.0,
            'active_clients': 0,
            'server_uptime': time.time(),
            'total_frames_served': 0,
            'camera_status': 'active',
            'resolution': f"{self.camera.width}x{self.camera.height}"
        }
        
        # Frame buffer for multiple clients
        self.frame_buffer = queue.Queue(maxsize=10)
        self.frame_interval = 1.0 / 30
        
        # Start frame distribution thread
        self.stop_frame_thread = threading.Event()
        self.frame_thread = threading.Thread(target=self._frame_distributor, daemon=True)
        self.frame_thread.start()
        
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>📷 Raspberry Pi Camera Stream</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    :root {
                        --primary: #4CAF50;
                        --secondary: #2196F3;
                        --dark: #121212;
                        --darker: #0a0a0a;
                        --dark-gray: #1a1a1a;
                        --gray: #2a2a2a;
                        --light-gray: #444;
                        --text: #f0f0f0;
                        --text-secondary: #aaa;
                    }
                    
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: 'Segoe UI', 'Roboto', sans-serif;
                        background: var(--dark);
                        color: var(--text);
                        line-height: 1.6;
                        min-height: 100vh;
                        overflow-x: hidden;
                    }
                    
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    
                    /* Header Styles */
                    .header {
                        text-align: center;
                        padding: 30px 20px;
                        background: linear-gradient(135deg, var(--darker) 0%, var(--dark-gray) 100%);
                        border-radius: 20px;
                        margin-bottom: 30px;
                        border: 1px solid var(--light-gray);
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .header::before {
                        content: '';
                        position: absolute;
                        top: -50%;
                        left: -50%;
                        width: 200%;
                        height: 200%;
                        background: radial-gradient(circle, rgba(76, 175, 80, 0.1) 0%, transparent 70%);
                    }
                    
                    .header h1 {
                        font-size: 2.8rem;
                        margin-bottom: 10px;
                        background: linear-gradient(90deg, var(--primary), var(--secondary));
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        position: relative;
                        z-index: 1;
                    }
                    
                    .header p {
                        font-size: 1.2rem;
                        color: var(--text-secondary);
                        margin-bottom: 20px;
                        position: relative;
                        z-index: 1;
                    }
                    
                    /* Status Bar */
                    .status-bar {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 15px;
                        margin-bottom: 30px;
                    }
                    
                    .status-card {
                        background: var(--dark-gray);
                        padding: 20px;
                        border-radius: 15px;
                        border-left: 4px solid var(--primary);
                        transition: transform 0.3s, box-shadow 0.3s;
                    }
                    
                    .status-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.2);
                    }
                    
                    .status-card i {
                        font-size: 1.5rem;
                        margin-right: 10px;
                        color: var(--primary);
                    }
                    
                    .status-card h3 {
                        font-size: 1rem;
                        color: var(--text-secondary);
                        margin-bottom: 8px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }
                    
                    .status-card .value {
                        font-size: 1.8rem;
                        font-weight: bold;
                        color: var(--text);
                    }
                    
                    .status-card .unit {
                        font-size: 1rem;
                        color: var(--text-secondary);
                    }
                    
                    /* Video Container */
                    .video-container {
                        background: var(--darker);
                        border-radius: 20px;
                        padding: 15px;
                        margin-bottom: 30px;
                        border: 1px solid var(--light-gray);
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .video-wrapper {
                        position: relative;
                        border-radius: 10px;
                        overflow: hidden;
                        background: #000;
                        aspect-ratio: 16/9;
                    }
                    
                    .video-wrapper img {
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                        transition: opacity 0.3s;
                    }
                    
                    .video-overlay {
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: rgba(0, 0, 0, 0.7);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        flex-direction: column;
                        opacity: 0;
                        transition: opacity 0.3s;
                    }
                    
                    .video-wrapper:hover .video-overlay {
                        opacity: 1;
                    }
                    
                    /* Connection Info */
                    .connection-info {
                        text-align: center;
                        padding: 20px;
                        background: var(--dark-gray);
                        border-radius: 15px;
                        margin-top: 30px;
                        border: 1px solid var(--light-gray);
                    }
                    
                    .connection-info p {
                        color: var(--text-secondary);
                        margin: 5px 0;
                    }
                    
                    .connection-info .ip-address {
                        color: var(--primary);
                        font-family: monospace;
                        font-size: 1.2rem;
                        background: rgba(0, 0, 0, 0.3);
                        padding: 5px 15px;
                        border-radius: 10px;
                        display: inline-block;
                        margin: 10px 0;
                    }
                    
                    /* Animations */
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(20px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    
                    .fade-in {
                        animation: fadeIn 0.6s ease-out;
                    }
                    
                    /* Responsive */
                    @media (max-width: 768px) {
                        .container {
                            padding: 10px;
                        }
                        
                        .header h1 {
                            font-size: 2rem;
                        }
                        
                        .status-bar {
                            grid-template-columns: 1fr;
                        }
                    }
                    
                    /* Loading spinner */
                    .spinner {
                        width: 40px;
                        height: 40px;
                        border: 4px solid var(--light-gray);
                        border-top: 4px solid var(--primary);
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                    }
                    
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <!-- Header -->
                    <header class="header fade-in">
                        <h1><i class="fas fa-camera"></i> Raspberry Pi Camera Stream</h1>
                        <p>Live streaming from Raspberry Pi Camera Module</p>
                    </header>
                    
                    <!-- Status Bar -->
                    <div class="status-bar fade-in" style="animation-delay: 0.2s">
                        <div class="status-card">
                            <h3><i class="fas fa-tachometer-alt"></i> Frame Rate</h3>
                            <div class="value" id="fps-value">25.0</div>
                            <span class="unit">FPS</span>
                        </div>
                        <div class="status-card">
                            <h3><i class="fas fa-thermometer-half"></i> CPU Temperature</h3>
                            <div class="value" id="temp-value">45.0</div>
                            <span class="unit">°C</span>
                        </div>
                        <div class="status-card">
                            <h3><i class="fas fa-video"></i> Resolution</h3>
                            <div class="value" id="res-value">640x480</div>
                            <span class="unit">pixels</span>
                        </div>
                        <div class="status-card">
                            <h3><i class="fas fa-users"></i> Viewers</h3>
                            <div class="value" id="client-count">0</div>
                            <span class="unit">connected</span>
                        </div>
                    </div>
                    
                    <!-- Video Stream -->
                    <div class="video-container fade-in" style="animation-delay: 0.4s">
                        <div class="video-wrapper">
                            <img id="video-feed" src="/video_feed" alt="Live Camera Stream">
                            <div class="video-overlay">
                                <div class="spinner"></div>
                                <p style="margin-top: 20px;">Camera stream loading...</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Connection Info -->
                    <div class="connection-info fade-in" style="animation-delay: 0.5s">
                        <p><i class="fas fa-wifi"></i> Raspberry Pi Streaming Server</p>
                        <p class="ip-address" id="ip-address">Loading server IP...</p>
                        <p><i class="fas fa-camera"></i> Camera Status: <span id="camera-status">Active</span></p>
                        <p><i class="fas fa-history"></i> Server Uptime: <span id="uptime">00:00:00</span></p>
                        <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.7;">
                            Auto-refreshes every 2 seconds | Frames served: <span id="frame-count">0</span>
                        </p>
                    </div>
                </div>
                
                <script>
                    // Get server IP
                    fetch('/server_info')
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById('ip-address').textContent = data.ip + ':' + data.port;
                            document.getElementById('res-value').textContent = data.resolution;
                        });
                    
                    // Format time
                    function formatTime(seconds) {
                        const hrs = Math.floor(seconds / 3600);
                        const mins = Math.floor((seconds % 3600) / 60);
                        const secs = Math.floor(seconds % 60);
                        return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                    }
                    
                    // Update stats
                    function updateStats() {
                        fetch('/stats')
                            .then(r => r.json())
                            .then(data => {
                                // Update status cards
                                document.getElementById('fps-value').textContent = data.fps.toFixed(1);
                                document.getElementById('temp-value').textContent = data.temperature.toFixed(1);
                                document.getElementById('client-count').textContent = data.active_clients;
                                document.getElementById('uptime').textContent = formatTime(data.server_uptime);
                                document.getElementById('frame-count').textContent = data.total_frames_served;
                                document.getElementById('camera-status').textContent = data.camera_status;
                            })
                            .catch(err => console.error('Error updating stats:', err));
                    }
                    
                    // Update video feed with error handling
                    const videoFeed = document.getElementById('video-feed');
                    videoFeed.onerror = function() {
                        this.onerror = null; // Prevent infinite loop
                        this.src = '/video_feed?' + new Date().getTime(); // Force reload
                    };
                    
                    // Initial update
                    updateStats();
                    
                    // Update every 2 seconds
                    setInterval(updateStats, 2000);
                </script>
            </body>
            </html>
            '''
        
        @self.app.route('/video_feed')
        def video_feed():
            def generate():
                try:
                    with self.client_lock:
                        if self.active_clients >= self.max_clients:
                            # Return overload frame
                            ret, buffer = cv2.imencode('.jpg', self._create_overload_frame())
                            if ret:
                                frame = buffer.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                            return
                        self.active_clients += 1
                        self.stream_info['active_clients'] = self.active_clients
                
                    while True:
                        try:
                            # Get frame from buffer with timeout
                            jpeg_data = self.frame_buffer.get(timeout=1.0)
                            self.stream_info['total_frames_served'] += 1
                            
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_data + b'\r\n')
                        except queue.Empty:
                            # Send placeholder if no frames
                            ret, buffer = cv2.imencode('.jpg', self._create_placeholder())
                            if ret:
                                frame = buffer.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    print(f"Stream error: {e}")
                finally:
                    with self.client_lock:
                        self.active_clients = max(0, self.active_clients - 1)
                        self.stream_info['active_clients'] = self.active_clients
            
            return Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame',
                          headers={
                              'Cache-Control': 'no-cache, no-store, must-revalidate',
                              'Pragma': 'no-cache',
                              'Expires': '0'
                          })
        
        @self.app.route('/stats')
        def get_stats():
            return jsonify({
                'fps': self.stream_info['fps'],
                'frame_count': self.camera.frame_count,
                'temperature': self.camera.temperature,
                'processing_time': self.stream_info['processing_time'],
                'active_clients': self.stream_info['active_clients'],
                'server_uptime': time.time() - self.stream_info['server_uptime'],
                'total_frames_served': self.stream_info['total_frames_served'],
                'camera_status': 'active' if self.camera.running else 'inactive',
                'resolution': self.stream_info['resolution']
            })
        
        @self.app.route('/server_info')
        def server_info():
            return jsonify({
                'ip': self._get_ip_address(),
                'port': self.port,
                'max_clients': self.max_clients,
                'status': 'running',
                'camera': 'ffplay',
                'resolution': self.stream_info['resolution']
            })
    
    def _frame_distributor(self):
        """Optimized frame distributor for multiple clients"""
        last_frame_time = 0
        fps_update_time = time.time()
        frame_counter = 0
        quality = 85
        
        while not self.stop_frame_thread.is_set():
            try:
                current_time = time.time()
                
                # Adjust frame rate and quality based on client count
                if self.active_clients > 10:
                    self.frame_interval = 1.0 / 20
                    quality = 70
                elif self.active_clients > 5:
                    self.frame_interval = 1.0 / 25
                    quality = 75
                else:
                    self.frame_interval = 1.0 / 30
                    quality = 85
                
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Get frame from camera
                frame = self.camera.get_frame()
                
                if frame is not None:
                    # Add timestamp overlay
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, timestamp, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add camera info
                    info_text = f"Pi Camera | {self.stream_info['resolution']} | {self.stream_info['fps']:.1f} FPS"
                    cv2.putText(frame, info_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # Add viewer count
                    viewers_text = f"Viewers: {self.active_clients}"
                    cv2.putText(frame, viewers_text, (frame.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Encode frame once for all clients
                    ret, buffer = cv2.imencode('.jpg', frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if ret:
                        jpeg_data = buffer.tobytes()
                        
                        # Put in buffer for clients
                        try:
                            if self.frame_buffer.full():
                                self.frame_buffer.get_nowait()
                            self.frame_buffer.put_nowait(jpeg_data)
                        except:
                            pass
                        
                        frame_counter += 1
                
                # Update FPS calculation
                if current_time - fps_update_time >= 1.0:
                    self.stream_info['fps'] = frame_counter / (current_time - fps_update_time)
                    self.stream_info['temperature'] = self.camera.temperature
                    self.stream_info['processing_time'] = 1000 / max(self.stream_info['fps'], 1)
                    frame_counter = 0
                    fps_update_time = current_time
                
                last_frame_time = current_time
                
            except Exception as e:
                print(f"Frame distributor error: {e}")
                time.sleep(0.1)
    
    def _create_placeholder(self):
        """Create placeholder frame"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(img.shape[0]):
            color = int(30 + (i / img.shape[0]) * 40)
            img[i, :] = (color, color, color)
        
        cv2.putText(img, "📷 PI CAMERA", (180, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (76, 175, 80), 2)
        cv2.putText(img, "Waiting for camera feed...", (160, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return img
    
    def _create_overload_frame(self):
        """Create frame when server is at max capacity"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Orange gradient background
        for i in range(img.shape[0]):
            red = int(50 + (i / img.shape[0]) * 100)
            img[i, :] = (0, 100, red)
        
        cv2.putText(img, "SERVER AT CAPACITY", (120, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        cv2.putText(img, f"Max viewers: {self.max_clients}", (180, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)
        
        cv2.putText(img, "Try again later", (220, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 200), 2)
        
        return img
    
    def run(self):
        """Start the streaming server"""
        # Get local IP
        ip = self._get_ip_address()
        
        # Start in background thread
        def start_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True, use_reloader=False)
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        print("\n" + "="*70)
        print("🚀 RASPBERRY PI CAMERA STREAMING SERVER")
        print("="*70)
        print(f"📡 On THIS computer, open:")
        print(f"   • http://localhost:{self.port}")
        print(f"   • http://{ip}:{self.port}")
        print(f"📱 On other devices, open:")
        print(f"   • http://{ip}:{self.port}")
        print("="*70)
        print("📷 Camera Information:")
        print(f"   • Resolution: {self.stream_info['resolution']}")
        print(f"   • Target FPS: 30")
        print(f"   • Source: ffplay -> /dev/video0")
        print("="*70)
        print("🔧 Features:")
        print("   • Live Pi camera feed")
        print("   • Adaptive quality/FPS based on viewers")
        print("   • Real-time stats display")
        print("   • Multi-client support")
        print("="*70)
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        return server_thread
    
    def stop(self):
        """Stop the streaming server"""
        self.stop_frame_thread.set()
        self.camera.stop()
    
    def _get_ip_address(self):
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))
            ip = s.getsockname()[0]
        except:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

# ==================== MAIN ====================
def run_camera_stream():
    """Run Pi camera streaming server"""
    print("🚀 Starting Raspberry Pi Camera Streaming Server...")
    
    # Start streaming server
    streamer = CameraStreamingServer(port=5000, max_clients=20)
    server_thread = streamer.run()
    
    try:
        # Keep server running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping camera streaming server...")
    
    streamer.stop()
    print("\n✅ Camera streaming stopped!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, 
                       help='Port for streaming server (default: 5000)')
    parser.add_argument('--clients', type=int, default=20,
                       help='Max concurrent clients (default: 20)')
    parser.add_argument('--resolution', type=str, default="640x480",
                       help='Camera resolution (default: 640x480)')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    run_camera_stream()