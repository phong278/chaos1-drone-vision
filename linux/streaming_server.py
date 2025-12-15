# streaming_server.py
from flask import Flask, Response, jsonify
import cv2
import threading
import time
import socket
import numpy as np
import queue
from datetime import datetime
import json

app = Flask(__name__)

class VideoStreamer:
    def __init__(self, port=5000, max_clients=20):
        self.port = port
        self.max_clients = max_clients
        self.active_clients = 0
        self.client_lock = threading.Lock()
        
        # Frame management
        self.latest_frame = None
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()
        self.is_streaming = False
        
        # Performance tracking
        self.detection_info = {
            'fps': 0.0,
            'frame_count': 0,
            'detections': [],
            'temperature': None,
            'processing_time': 0,
            'active_clients': 0,
            'server_uptime': time.time(),
            'total_frames_served': 0
        }
        
        # Frame buffer for multiple clients
        self.frame_buffer = queue.Queue(maxsize=10)
        self.frame_interval = 1.0 / 30  # Max 30 FPS for streaming
        
        # Start frame distribution thread
        self.stop_frame_thread = threading.Event()
        self.frame_thread = threading.Thread(target=self._frame_distributor, daemon=True)
        self.frame_thread.start()
        
        # Setup routes
        self.setup_routes()
    
    def setup_routes(self):
        @app.route('/')
        def index():
            return '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Pi 4 Drone Camera - Live Stream</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    :root {
                        --primary: #00ff88;
                        --secondary: #0088ff;
                        --dark: #0a0a0a;
                        --darker: #050505;
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
                        background: radial-gradient(circle, rgba(0, 255, 136, 0.1) 0%, transparent 70%);
                        animation: pulse 10s infinite alternate;
                    }
                    
                    @keyframes pulse {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
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
                        box-shadow: 0 5px 15px rgba(0, 255, 136, 0.2);
                    }
                    
                    .status-card.warning {
                        border-left-color: #ff9900;
                    }
                    
                    .status-card.danger {
                        border-left-color: #ff3333;
                    }
                    
                    .status-card i {
                        font-size: 1.5rem;
                        margin-right: 10px;
                        color: var(--primary);
                    }
                    
                    .status-card.warning i { color: #ff9900; }
                    .status-card.danger i { color: #ff3333; }
                    
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
                    
                    /* Detections Panel */
                    .detections-panel {
                        background: var(--dark-gray);
                        border-radius: 20px;
                        padding: 25px;
                        margin-bottom: 30px;
                        border: 1px solid var(--light-gray);
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                    }
                    
                    .panel-header {
                        display: flex;
                        align-items: center;
                        margin-bottom: 20px;
                    }
                    
                    .panel-header h2 {
                        font-size: 1.5rem;
                        color: var(--text);
                        margin-right: auto;
                    }
                    
                    .detection-count {
                        background: var(--primary);
                        color: var(--dark);
                        padding: 5px 15px;
                        border-radius: 20px;
                        font-weight: bold;
                        font-size: 0.9rem;
                    }
                    
                    .detections-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                        gap: 15px;
                        margin-bottom: 20px;
                    }
                    
                    .detection-card {
                        background: var(--gray);
                        padding: 15px;
                        border-radius: 10px;
                        border: 1px solid var(--light-gray);
                        transition: all 0.3s;
                    }
                    
                    .detection-card:hover {
                        border-color: var(--primary);
                        transform: translateY(-3px);
                    }
                    
                    .detection-label {
                        display: flex;
                        align-items: center;
                        margin-bottom: 10px;
                    }
                    
                    .detection-label .icon {
                        width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        background: var(--primary);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 10px;
                    }
                    
                    .detection-label .icon.person { background: #ff3333; }
                    .detection-label .icon.vehicle { background: #3388ff; }
                    .detection-label .icon.animal { background: #33ff88; }
                    .detection-label .icon.object { background: #ffaa33; }
                    
                    .detection-label span {
                        font-weight: bold;
                        font-size: 1.1rem;
                    }
                    
                    .confidence-bar {
                        height: 6px;
                        background: var(--light-gray);
                        border-radius: 3px;
                        margin: 10px 0;
                        overflow: hidden;
                    }
                    
                    .confidence-fill {
                        height: 100%;
                        background: var(--primary);
                        border-radius: 3px;
                        transition: width 1s ease-in-out;
                    }
                    
                    .detection-meta {
                        font-size: 0.85rem;
                        color: var(--text-secondary);
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 5px;
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
                        
                        .detections-grid {
                            grid-template-columns: 1fr;
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
                        <h1><i class="fas fa-drone"></i> Pi 4 Drone Camera</h1>
                        <p>Real-time object detection & live streaming from Raspberry Pi 4</p>
                        <div class="connection-info">
                            <p><i class="fas fa-wifi"></i> Live Streaming Active</p>
                            <p class="ip-address" id="ip-address">Loading...</p>
                            <p><i class="fas fa-users"></i> Connected Clients: <span id="client-count">0</span>/<span id="max-clients">20</span></p>
                        </div>
                    </header>
                    
                    <!-- Status Bar -->
                    <div class="status-bar fade-in" style="animation-delay: 0.2s">
                        <div class="status-card">
                            <h3><i class="fas fa-tachometer-alt"></i> Frame Rate</h3>
                            <div class="value" id="fps-value">0.0</div>
                            <span class="unit">FPS</span>
                        </div>
                        <div class="status-card">
                            <h3><i class="fas fa-microchip"></i> CPU Temperature</h3>
                            <div class="value" id="temp-value">--</div>
                            <span class="unit">°C</span>
                        </div>
                        <div class="status-card">
                            <h3><i class="fas fa-clock"></i> Processing Time</h3>
                            <div class="value" id="process-value">0</div>
                            <span class="unit">ms</span>
                        </div>
                        <div class="status-card">
                            <h3><i class="fas fa-hdd"></i> Frames Served</h3>
                            <div class="value" id="frame-count">0</div>
                            <span class="unit">total</span>
                        </div>
                    </div>
                    
                    <!-- Video Stream -->
                    <div class="video-container fade-in" style="animation-delay: 0.4s">
                        <div class="video-wrapper">
                            <img id="video-feed" src="/video_feed" alt="Live Stream">
                            <div class="video-overlay">
                                <div class="spinner"></div>
                                <p style="margin-top: 20px;">Live stream loading...</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Detections Panel -->
                    <div class="detections-panel fade-in" style="animation-delay: 0.6s">
                        <div class="panel-header">
                            <h2><i class="fas fa-bullseye"></i> Active Detections</h2>
                            <div class="detection-count" id="detection-count">0 objects</div>
                        </div>
                        <div class="detections-grid" id="detections-grid">
                            <!-- Detections will appear here -->
                            <div class="empty-state" style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-secondary);">
                                <i class="fas fa-search" style="font-size: 3rem; margin-bottom: 20px; opacity: 0.5;"></i>
                                <h3>No objects detected</h3>
                                <p>Waiting for detection data...</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Footer -->
                    <footer class="connection-info fade-in" style="animation-delay: 0.8s">
                        <p><i class="fas fa-satellite-dish"></i> Raspberry Pi 4 Detection System</p>
                        <p><i class="fas fa-history"></i> Server Uptime: <span id="uptime">00:00:00</span></p>
                        <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.7;">
                            Auto-refreshes every 5 seconds | Stream optimized for multiple clients
                        </p>
                    </footer>
                </div>
                
                <script>
                    // Get server IP
                    fetch('/server_info')
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById('ip-address').textContent = data.ip + ':' + data.port;
                            document.getElementById('max-clients').textContent = data.max_clients;
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
                                document.getElementById('temp-value').textContent = data.temperature ? data.temperature.toFixed(1) : '--';
                                document.getElementById('process-value').textContent = data.processing_time.toFixed(1);
                                document.getElementById('frame-count').textContent = data.frame_count;
                                document.getElementById('client-count').textContent = data.active_clients;
                                document.getElementById('uptime').textContent = formatTime(data.server_uptime);
                                
                                // Update temperature card color
                                const tempCard = document.querySelector('.status-card:nth-child(2)');
                                if (data.temperature) {
                                    tempCard.classList.remove('warning', 'danger');
                                    if (data.temperature > 70) tempCard.classList.add('warning');
                                    if (data.temperature > 80) tempCard.classList.add('danger');
                                }
                                
                                // Update detections
                                const grid = document.getElementById('detections-grid');
                                const countElem = document.getElementById('detection-count');
                                
                                if (data.detections && data.detections.length > 0) {
                                    countElem.textContent = data.detections.length + ' object' + (data.detections.length !== 1 ? 's' : '');
                                    
                                    let html = '';
                                    data.detections.forEach((det, index) => {
                                        // Determine icon class based on label
                                        let iconClass = 'object';
                                        if (det.label.toLowerCase().includes('person')) iconClass = 'person';
                                        else if (det.label.toLowerCase().includes('car') || det.label.toLowerCase().includes('bus') || det.label.toLowerCase().includes('truck')) iconClass = 'vehicle';
                                        else if (det.label.toLowerCase().includes('bird') || det.label.toLowerCase().includes('cat') || det.label.toLowerCase().includes('dog')) iconClass = 'animal';
                                        
                                        // Determine icon
                                        let icon = 'fa-cube';
                                        switch(iconClass) {
                                            case 'person': icon = 'fa-user'; break;
                                            case 'vehicle': icon = 'fa-car'; break;
                                            case 'animal': icon = 'fa-paw'; break;
                                            default: icon = 'fa-cube';
                                        }
                                        
                                        html += `
                                            <div class="detection-card fade-in" style="animation-delay: ${index * 0.1}s">
                                                <div class="detection-label">
                                                    <div class="icon ${iconClass}">
                                                        <i class="fas ${icon}"></i>
                                                    </div>
                                                    <span>${det.label.charAt(0).toUpperCase() + det.label.slice(1)}</span>
                                                </div>
                                                <div class="confidence-bar">
                                                    <div class="confidence-fill" style="width: ${det.confidence * 100}%"></div>
                                                </div>
                                                <div class="detection-meta">
                                                    <div><i class="fas fa-percentage"></i> ${(det.confidence * 100).toFixed(1)}%</div>
                                                    <div><i class="fas fa-expand-arrows-alt"></i> ${det.bbox[2] - det.bbox[0]}×${det.bbox[3] - det.bbox[1]}</div>
                                                    <div><i class="fas fa-map-marker-alt"></i> [${det.bbox[0]}, ${det.bbox[1]}]</div>
                                                    <div><i class="fas fa-history"></i> ${data.fps.toFixed(0)} FPS</div>
                                                </div>
                                            </div>
                                        `;
                                    });
                                    grid.innerHTML = html;
                                } else {
                                    countElem.textContent = '0 objects';
                                    grid.innerHTML = `
                                        <div class="empty-state" style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-secondary);">
                                            <i class="fas fa-search" style="font-size: 3rem; margin-bottom: 20px; opacity: 0.5;"></i>
                                            <h3>No objects detected</h3>
                                            <p>Waiting for detection data...</p>
                                        </div>
                                    `;
                                }
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
                    
                    // Auto-refresh page every 5 minutes
                    setTimeout(() => {
                        window.location.reload();
                    }, 300000);
                </script>
            </body>
            </html>
            '''
        
        @app.route('/video_feed')
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
                        self.detection_info['active_clients'] = self.active_clients
                
                    while True:
                        try:
                            # Get frame from buffer with timeout
                            jpeg_data = self.frame_buffer.get(timeout=1.0)
                            self.detection_info['total_frames_served'] += 1
                            
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
                        self.detection_info['active_clients'] = self.active_clients
            
            return Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame',
                          headers={
                              'Cache-Control': 'no-cache, no-store, must-revalidate',
                              'Pragma': 'no-cache',
                              'Expires': '0'
                          })
        
        @app.route('/stats')
        def get_stats():
            with self.frame_lock:
                return jsonify({
                    'fps': self.detection_info['fps'],
                    'frame_count': self.detection_info['frame_count'],
                    'detections': self.detection_info['detections'],
                    'temperature': self.detection_info['temperature'],
                    'processing_time': self.detection_info.get('processing_time', 0),
                    'active_clients': self.detection_info.get('active_clients', 0),
                    'server_uptime': time.time() - self.detection_info['server_uptime'],
                    'total_frames_served': self.detection_info.get('total_frames_served', 0)
                })
        
        @app.route('/server_info')
        def server_info():
            return jsonify({
                'ip': self._get_ip_address(),
                'port': self.port,
                'max_clients': self.max_clients,
                'status': 'running'
            })
    
    def _frame_distributor(self):
        """Optimized frame distributor for multiple clients"""
        last_frame_time = 0
        quality = 85  # Start with high quality
        
        while not self.stop_frame_thread.is_set():
            try:
                current_time = time.time()
                
                # Adjust frame rate based on client count
                if self.active_clients > 10:
                    self.frame_interval = 1.0 / 20  # 20 FPS max for many clients
                    quality = 70
                elif self.active_clients > 5:
                    self.frame_interval = 1.0 / 25  # 25 FPS for moderate clients
                    quality = 75
                else:
                    self.frame_interval = 1.0 / 30  # 30 FPS for few clients
                    quality = 85
                
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                
                with self.frame_lock:
                    if self.latest_frame is None:
                        continue
                    
                    # Encode frame once for all clients
                    ret, buffer = cv2.imencode('.jpg', self.latest_frame, 
                                              [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if ret:
                        jpeg_data = buffer.tobytes()
                        
                        # Put in buffer for clients
                        try:
                            if self.frame_buffer.full():
                                self.frame_buffer.get_nowait()  # Remove oldest frame
                            self.frame_buffer.put_nowait(jpeg_data)
                        except:
                            pass
                
                last_frame_time = current_time
                
            except Exception as e:
                print(f"Frame distributor error: {e}")
                time.sleep(0.1)
    
    def update_frame(self, frame, detections, fps=0.0, frame_count=0, temperature=None, processing_time=0):
        """Update the latest frame with detections - optimized for performance"""
        frame_with_overlay = frame.copy()
        
        # Draw detections (only if we have them)
        detection_list = []
        if detections:
            for det in detections.values():
                label = det['label']
                conf = det['confidence']
                x1, y1, x2, y2 = det['bbox']
                
                # Choose color based on label
                if 'person' in label.lower():
                    color = (0, 100, 255)  # Orange-red
                elif 'car' in label.lower() or 'vehicle' in label.lower():
                    color = (255, 100, 0)  # Blue
                elif 'animal' in label.lower():
                    color = (100, 255, 0)  # Green
                else:
                    color = (255, 255, 100)  # Light yellow
                
                # Draw bounding box
                cv2.rectangle(frame_with_overlay, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background for readability
                label_text = f"{label[:15]}: {conf:.0%}"
                (text_width, text_height), _ = cv2.getTextSize(label_text, 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Draw label background
                cv2.rectangle(frame_with_overlay, 
                             (x1, y1 - text_height - 10),
                             (x1 + text_width + 5, y1),
                             color, -1)
                
                # Draw label text
                cv2.putText(frame_with_overlay, label_text, 
                           (x1 + 3, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (255, 255, 255), 1)
                
                # Add to detection list for JSON
                detection_list.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        # Add info overlay with better styling
        overlay_height = 90 if temperature else 60
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (20, 20, 20)  # Dark gray background
        
        # FPS and frame count
        fps_text = f"FPS: {fps:5.1f} | Frame: {frame_count}"
        cv2.putText(overlay, fps_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Processing time
        proc_text = f"Process: {processing_time:5.1f}ms"
        cv2.putText(overlay, proc_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if processing_time < 50 else (0, 255, 255) if processing_time < 100 else (0, 0, 255), 
                   2)
        
        # Temperature if available
        if temperature:
            temp_color = (0, 255, 0)  # Green
            if temperature > 70:
                temp_color = (0, 255, 255)  # Yellow
            if temperature > 80:
                temp_color = (0, 0, 255)  # Red
            
            temp_text = f"Temp: {temperature:5.1f}°C"
            cv2.putText(overlay, temp_text, (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, temp_color, 2)
        
        # Blend overlay onto frame
        alpha = 0.7
        frame_with_overlay[:overlay_height] = cv2.addWeighted(
            overlay, alpha, frame_with_overlay[:overlay_height], 1 - alpha, 0)
        
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
    
    def _create_placeholder(self):
        """Create a placeholder frame when no camera feed"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(img.shape[0]):
            color = int(30 + (i / img.shape[0]) * 40)
            img[i, :] = (color, color, color)
        
        # Text with shadow
        shadow_color = (10, 10, 10)
        text_color = (0, 200, 255)
        
        cv2.putText(img, "Waiting for camera feed...", (102, 242),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, shadow_color, 3)
        cv2.putText(img, "Waiting for camera feed...", (100, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        cv2.putText(img, "Pi 4 Drone Detection System", (92, 282),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow_color, 3)
        cv2.putText(img, "Pi 4 Drone Detection System", (90, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
        
        # Add spinner animation
        center = (320, 380)
        radius = 30
        angle = int(time.time() * 180) % 360
        end_point = (
            int(center[0] + radius * np.cos(np.radians(angle))),
            int(center[1] + radius * np.sin(np.radians(angle)))
        )
        cv2.circle(img, center, radius, (100, 100, 255), 2)
        cv2.line(img, center, end_point, (0, 255, 255), 3)
        
        return img
    
    def _create_overload_frame(self):
        """Create frame when server is at max capacity"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Red gradient background
        for i in range(img.shape[0]):
            red = int(50 + (i / img.shape[0]) * 100)
            img[i, :] = (0, 0, red)
        
        cv2.putText(img, "SERVER AT CAPACITY", (120, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        cv2.putText(img, "Too many clients connected", (100, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
        
        cv2.putText(img, f"Max: {self.max_clients} clients", (180, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)
        
        return img
    
    def run(self):
        """Start the streaming server with production-ready settings"""
        self.is_streaming = True
        
        # Get IP address
        ip = self._get_ip_address()
        
        # Start server with better settings
        def run_server():
            # Production settings for Flask dev server (still recommend Gunicorn for >10 clients)
            app.run(host='0.0.0.0', 
                   port=self.port, 
                   debug=False, 
                   threaded=True,
                   use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        print("\n" + "="*70)
        print("OPTIMIZED STREAMING SERVER STARTED")
        print("="*70)
        print(f"Local Access:    http://localhost:{self.port}")
        print(f"Network Access:  http://{ip}:{self.port}")
        print(f"Max Clients:     {self.max_clients}")
        print("="*70)
        
        print("="*70 + "\n")
        
        return server_thread
    
    def stop(self):
        """Stop the streaming server"""
        self.stop_frame_thread.set()
        self.is_streaming = False
    
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
    print("Testing optimized streaming server...")
    streamer = VideoStreamer(port=5000, max_clients=20)
    streamer.run()
    
    # Test with dummy frames
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "Test Stream - Optimized", (150, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    try:
        while True:
            streamer.update_frame(test_img, {}, fps=30.0, frame_count=123, temperature=45.5, processing_time=25.3)
            time.sleep(0.033)  # ~30 FPS
    except KeyboardInterrupt:
        streamer.stop()
        print("\nServer stopped.")