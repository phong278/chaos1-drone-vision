#!/usr/bin/env python3
"""
Fixed Camera Streamer - Working MJPEG streaming
"""

import subprocess
import time
import threading
from flask import Flask, Response
import signal
import sys

print("📷 FIXED CAMERA STREAMER")
print("="*60)

app = Flask(__name__)

# Global variables
camera_process = None
camera_running = False
frame_buffer = []
buffer_lock = threading.Lock()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def find_jpeg_boundaries(data):
    """Find JPEG frame boundaries in stream"""
    frames = []
    start = 0
    
    while True:
        # Find JPEG start marker (FFD8)
        jpeg_start = data.find(b'\xff\xd8', start)
        if jpeg_start == -1:
            break
        
        # Find JPEG end marker (FFD9)
        jpeg_end = data.find(b'\xff\xd9', jpeg_start + 2)
        if jpeg_end == -1:
            break
        
        # Extract complete JPEG frame
        frame = data[jpeg_start:jpeg_end + 2]
        frames.append(frame)
        start = jpeg_end + 2
    
    return frames

def read_camera_stream():
    """Read camera stream and buffer frames"""
    global camera_process, frame_buffer
    
    log("📡 Starting frame reader thread...")
    buffer = b''
    
    while camera_running and camera_process:
        try:
            # Read chunk from ffmpeg
            chunk = camera_process.stdout.read(4096)
            
            if not chunk:
                log("⚠️ No more data from camera")
                break
            
            buffer += chunk
            
            # Extract complete JPEG frames
            frames = find_jpeg_boundaries(buffer)
            
            if frames:
                with buffer_lock:
                    # Keep only last 10 frames
                    frame_buffer.extend(frames)
                    frame_buffer = frame_buffer[-10:]
                
                # Remove processed frames from buffer
                last_frame = frames[-1]
                last_pos = buffer.rfind(last_frame) + len(last_frame)
                buffer = buffer[last_pos:]
                
        except Exception as e:
            log(f"❌ Reader error: {e}")
            break
    
    log("📡 Frame reader stopped")

def start_camera():
    global camera_process, camera_running, frame_buffer
    
    if camera_running and camera_process and camera_process.poll() is None:
        return True
    
    if camera_process:
        stop_camera()
    
    log("🚀 Starting camera...")
    
    cmd = [
        'ffmpeg',
        '-f', 'v4l2',
        '-input_format', 'mjpeg',
        '-video_size', '640x480',
        '-framerate', '15',
        '-i', '/dev/video0',
        '-f', 'mjpeg',
        '-q:v', '5',  # Better quality
        '-huffman', 'optimal',
        '-'
    ]
    
    try:
        log(f"Running: {' '.join(cmd)}")
        
        camera_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0  # No buffering
        )
        
        # Start stderr reader
        def read_stderr():
            for line in camera_process.stderr:
                line_str = line.decode().strip()
                if not line_str.startswith('frame='):
                    log(f"FFMPEG: {line_str}")
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        # Wait for startup
        time.sleep(2)
        
        if camera_process.poll() is None:
            camera_running = True
            frame_buffer = []
            
            # Start frame reader thread
            reader_thread = threading.Thread(target=read_camera_stream, daemon=True)
            reader_thread.start()
            
            log("✅ Camera started successfully")
            return True
        else:
            log("❌ Camera failed to start")
            return False
            
    except Exception as e:
        log(f"❌ Error starting camera: {e}")
        return False

def stop_camera():
    global camera_process, camera_running
    if camera_process:
        log("📴 Stopping camera...")
        camera_running = False
        try:
            camera_process.terminate()
            camera_process.wait(timeout=2)
        except:
            try:
                camera_process.kill()
            except:
                pass
        camera_process = None

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
    <p class="subtitle">Live stream from /dev/video0</p>
    
    <div class="video-container">
        <div id="loading">Loading camera feed...</div>
        <img id="video-feed" style="display:none;" alt="Live Camera Feed">
    </div>
    
    <div class="status">
        <span class="live-indicator"></span>
        Status: <span id="status">Connecting...</span> | 
        FPS: <span id="fps">--</span> | 
        Time: <span id="timestamp">--:--:--</span>
    </div>
    
    <div class="controls">
        <button onclick="reloadStream()">🔄 Reload Stream</button>
        <button onclick="location.reload()">↻ Refresh Page</button>
    </div>
    
    <script>
        const video = document.getElementById('video-feed');
        const loading = document.getElementById('loading');
        const statusSpan = document.getElementById('status');
        const fpsSpan = document.getElementById('fps');
        const timestamp = document.getElementById('timestamp');
        
        let frameCount = 0;
        let lastFpsUpdate = Date.now();
        let streamLoaded = false;
        
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
            if (!streamLoaded) {
                console.log('First frame loaded');
                streamLoaded = true;
                loading.style.display = 'none';
                video.style.display = 'block';
                updateStatus('Live', '#4CAF50');
            }
            frameCount++;
            updateFps();
        };
        
        video.onerror = function(e) {
            console.error('Video error:', e);
            updateStatus('Error - Reconnecting...', '#ff4444');
            loading.style.display = 'block';
            loading.textContent = 'Connection lost. Reconnecting...';
            
            setTimeout(() => {
                reloadStream();
            }, 2000);
        };
        
        function reloadStream() {
            console.log('Reloading stream...');
            updateStatus('Reloading...', '#ffaa00');
            streamLoaded = false;
            video.src = '/stream.mjpg?t=' + Date.now();
        }
        
        // Update time every second
        setInterval(updateTime, 1000);
        updateTime();
        
        // Start stream
        console.log('Starting stream...');
        video.src = '/stream.mjpg';
        
        // Retry if not loaded after 10 seconds
        setTimeout(() => {
            if (!streamLoaded) {
                console.log('Stream timeout, retrying...');
                reloadStream();
            }
        }, 10000);
        
        // Keep-alive: reload every 60 seconds
        setInterval(() => {
            if (streamLoaded) {
                console.log('Keep-alive: refreshing stream');
                reloadStream();
            }
        }, 60000);
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
        if not camera_running:
            log("Starting camera for client")
            if not start_camera():
                log("Failed to start camera")
                return
            time.sleep(2)  # Wait for frames
        
        last_frame_idx = 0
        
        try:
            while True:
                with buffer_lock:
                    if len(frame_buffer) > last_frame_idx:
                        # Get new frame
                        frame = frame_buffer[-1]
                        last_frame_idx = len(frame_buffer)
                    else:
                        frame = None
                
                if frame:
                    # Send frame with proper MJPEG multipart format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(frame)).encode() + b'\r\n'
                           b'\r\n' + frame + b'\r\n')
                else:
                    # No new frame, wait a bit
                    time.sleep(0.01)
                    
        except GeneratorExit:
            log("🎥 Client disconnected")
        except Exception as e:
            log(f"❌ Stream error: {e}")
    
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'close'
        }
    )

@app.route('/status')
def status():
    with buffer_lock:
        buffer_size = len(frame_buffer)
    
    return {
        'camera_running': camera_running,
        'camera_alive': camera_process.poll() is None if camera_process else False,
        'frames_buffered': buffer_size
    }

def signal_handler(sig, frame):
    log("\n🛑 Stopping...")
    stop_camera()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("🚀 STARTING FIXED CAMERA STREAMER")
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