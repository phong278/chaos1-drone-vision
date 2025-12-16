#!/usr/bin/env python3
"""
Fixed Camera Streamer with Direct OpenCV Capture
"""

import cv2
import time
import threading
from flask import Flask, Response
import signal
import sys

print("📷 OPENCV CAMERA STREAMER")
print("="*60)

app = Flask(__name__)

# Global variables
camera = None
camera_lock = threading.Lock()
output_frame = None
frame_lock = threading.Lock()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def capture_frames():
    """Capture frames from camera in background thread"""
    global camera, output_frame
    
    log("📡 Starting frame capture thread...")
    
    while True:
        with camera_lock:
            if camera is None:
                break
            
            ret, frame = camera.read()
            
            if not ret:
                log("⚠️ Failed to read frame")
                time.sleep(0.1)
                continue
        
        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        if ret:
            with frame_lock:
                output_frame = jpeg.tobytes()
        
        time.sleep(0.033)  # ~30 FPS
    
    log("📡 Frame capture stopped")

def start_camera():
    global camera
    
    log("🚀 Starting camera...")
    
    with camera_lock:
        if camera is not None:
            return True
        
        # Open camera with OpenCV
        camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        if not camera.isOpened():
            log("❌ Failed to open camera")
            camera = None
            return False
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual properties
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = camera.get(cv2.CAP_PROP_FPS)
        
        log(f"✅ Camera opened: {width}x{height} @ {fps:.1f} FPS")
    
    # Start capture thread
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()
    
    # Wait for first frame
    for i in range(50):  # Wait up to 5 seconds
        time.sleep(0.1)
        with frame_lock:
            if output_frame is not None:
                log("✅ First frame captured")
                return True
    
    log("⚠️ No frames captured yet")
    return True

def stop_camera():
    global camera
    log("📴 Stopping camera...")
    
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

HTML = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pi Camera Stream</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #0a0a0a;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        
        h1 {
            color: #4CAF50;
            margin-bottom: 5px;
        }
        
        .subtitle {
            color: #aaa;
            margin-bottom: 20px;
        }
        
        .video-container {
            max-width: 800px;
            margin: 20px auto;
            background: black;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #333;
            position: relative;
            min-height: 480px;
        }
        
        #video-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #ffaa00;
            font-size: 18px;
            background: rgba(0,0,0,0.8);
            padding: 20px;
            border-radius: 10px;
        }
        
        .status {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 20px auto;
            max-width: 800px;
            font-family: monospace;
        }
        
        .controls {
            margin: 20px;
        }
        
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 0 10px;
        }
        
        button:hover {
            background: #45a049;
        }
        
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #4CAF50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
    </style>
</head>
<body>
    <h1>📷 Raspberry Pi Camera Stream</h1>
    <p class="subtitle">Live OpenCV stream from /dev/video0</p>
    
    <div class="video-container">
        <div id="loading">Loading camera feed...</div>
        <img id="video-feed" style="display:none;" alt="Live Camera Feed">
    </div>
    
    <div class="status">
        <span class="live-indicator"></span>
        Status: <span id="status">Connecting...</span> | 
        FPS: <span id="fps">--</span> | 
        Frames: <span id="frame-count">0</span> | 
        Time: <span id="timestamp">--:--:--</span>
    </div>
    
    <div class="controls">
        <button onclick="reloadStream()">🔄 Reload Stream</button>
        <button onclick="location.reload()">↻ Refresh Page</button>
        <button onclick="checkStatus()">📊 Check Status</button>
    </div>
    
    <div id="debug" style="background: rgba(255,255,255,0.1); padding: 10px; margin: 20px auto; max-width: 800px; font-family: monospace; font-size: 12px; display: none;">
        Debug: <span id="debug-msg">--</span>
    </div>
    
    <script>
        const video = document.getElementById('video-feed');
        const loading = document.getElementById('loading');
        const statusSpan = document.getElementById('status');
        const fpsSpan = document.getElementById('fps');
        const frameCountSpan = document.getElementById('frame-count');
        const timestamp = document.getElementById('timestamp');
        const debugDiv = document.getElementById('debug');
        const debugMsg = document.getElementById('debug-msg');
        
        let frameCount = 0;
        let totalFrames = 0;
        let lastFpsUpdate = Date.now();
        let streamLoaded = false;
        let retryCount = 0;
        
        function debug(msg) {
            console.log(msg);
            debugMsg.textContent = msg;
            debugDiv.style.display = 'block';
        }
        
        function updateTime() {
            const now = new Date();
            timestamp.textContent = now.toLocaleTimeString();
        }
        
        function updateStatus(text, color) {
            statusSpan.textContent = text;
            statusSpan.style.color = color || 'white';
        }
        
        function updateFps() {
            const now = Date.now();
            const elapsed = (now - lastFpsUpdate) / 1000;
            if (elapsed >= 1) {
                const currentFps = (frameCount / elapsed).toFixed(1);
                fpsSpan.textContent = currentFps;
                frameCount = 0;
                lastFpsUpdate = now;
            }
        }
        
        video.onload = function() {
            debug('Frame loaded');
            
            if (!streamLoaded) {
                console.log('First frame loaded successfully');
                streamLoaded = true;
                loading.style.display = 'none';
                video.style.display = 'block';
                updateStatus('Live', '#4CAF50');
                retryCount = 0;
            }
            
            frameCount++;
            totalFrames++;
            frameCountSpan.textContent = totalFrames;
            updateFps();
        };
        
        video.onerror = function(e) {
            console.error('Video error:', e);
            debug('Error loading frame');
            updateStatus('Error - Retrying...', '#ff4444');
            
            retryCount++;
            if (retryCount < 10) {
                setTimeout(() => {
                    debug('Retry ' + retryCount);
                    reloadStream();
                }, 2000);
            } else {
                updateStatus('Failed after 10 retries', '#ff0000');
                loading.textContent = 'Stream failed. Please refresh page.';
            }
        };
        
        function reloadStream() {
            console.log('Reloading stream...');
            debug('Reloading stream...');
            updateStatus('Reloading...', '#ffaa00');
            streamLoaded = false;
            video.src = '/stream.mjpg?t=' + Date.now();
        }
        
        function checkStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    debug('Camera: ' + (data.camera_running ? 'ON' : 'OFF') + 
                          ' | Frames: ' + data.frames_captured);
                })
                .catch(e => debug('Status check failed: ' + e));
        }
        
        // Update time every second
        setInterval(updateTime, 1000);
        updateTime();
        
        // Start stream
        console.log('Starting stream...');
        debug('Initializing...');
        video.src = '/stream.mjpg';
        
        // Retry if not loaded after 5 seconds
        setTimeout(() => {
            if (!streamLoaded) {
                console.log('Stream timeout, retrying...');
                debug('Timeout, retrying...');
                reloadStream();
            }
        }, 5000);
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return HTML

@app.route('/stream.mjpg')
def stream():
    def generate():
        log("🎥 Client connected to stream")
        
        # Ensure camera is running
        if camera is None:
            log("Starting camera for client")
            if not start_camera():
                log("Failed to start camera")
                return
        
        frame_num = 0
        
        try:
            while True:
                # Get current frame
                with frame_lock:
                    if output_frame is None:
                        time.sleep(0.01)
                        continue
                    
                    frame_bytes = output_frame
                
                # Send frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
                
                frame_num += 1
                
                if frame_num % 100 == 0:
                    log(f"📡 Sent {frame_num} frames to client")
                
                time.sleep(0.033)  # ~30 FPS
                
        except GeneratorExit:
            log(f"🎥 Client disconnected (sent {frame_num} frames)")
        except Exception as e:
            log(f"❌ Stream error: {e}")
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'close',
            'X-Accel-Buffering': 'no'
        }
    )

frames_captured = 0

@app.route('/status')
def status():
    global frames_captured
    
    with frame_lock:
        has_frame = output_frame is not None
        if has_frame:
            frames_captured += 1
    
    return {
        'camera_running': camera is not None,
        'has_frame': has_frame,
        'frames_captured': frames_captured
    }

def signal_handler(sig, frame):
    log("\n🛑 Stopping...")
    stop_camera()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("🚀 STARTING OPENCV CAMERA STREAMER")
    print("="*70)
    
    # Get IP
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
    except:
        local_ip = '127.0.0.1'
    
    print(f"\n🌐 Open in browser:")
    print(f"   http://localhost:5000")
    print(f"   http://{local_ip}:5000")
    print("\n🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Start camera
    start_camera()
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)