#!/usr/bin/env python3
"""
Working Camera Streamer - Fixed MJPEG streaming
"""

import subprocess
import time
import threading
from flask import Flask, Response
import signal
import sys
import os

print("📷 WORKING CAMERA STREAMER")
print("="*60)

app = Flask(__name__)

# Global variables
camera_process = None
camera_running = False

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def start_camera():
    global camera_process, camera_running
    
    if camera_running and camera_process and camera_process.poll() is None:
        return True
    
    # Stop existing process
    if camera_process:
        stop_camera()
    
    log("🚀 Starting camera...")
    
    # Simpler command - let ffmpeg handle more
    cmd = [
        'ffmpeg',
        '-f', 'v4l2',
        '-input_format', 'mjpeg',
        '-video_size', '640x480',
        '-framerate', '10',  # Lower for stability
        '-i', '/dev/video0',
        '-f', 'mjpeg',
        '-q:v', '8',
        '-vsync', 'vfr',  # Variable framerate
        '-fflags', 'nobuffer',  # Reduce buffering
        '-flush_packets', '1',  # Flush packets immediately
        '-'
    ]
    
    try:
        log(f"Running: {' '.join(cmd)}")
        
        camera_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**6,
            close_fds=True
        )
        
        # Start stderr reader
        def read_stderr():
            while True:
                line = camera_process.stderr.readline()
                if line:
                    line_str = line.decode().strip()
                    # Only show important messages
                    if not line_str.startswith('frame='):
                        log(f"FFMPEG: {line_str}")
                elif camera_process.poll() is not None:
                    break
                time.sleep(0.1)
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        # Give it time to start
        time.sleep(3)
        
        if camera_process.poll() is None:
            camera_running = True
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
        try:
            camera_process.terminate()
            camera_process.wait(timeout=2)
        except:
            try:
                camera_process.kill()
                camera_process.wait()
            except:
                pass
        camera_process = None
    camera_running = False

# HTML
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
        }
        
        #video-feed {
            width: 100%;
            height: auto;
            display: block;
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
        
        #loading {
            color: #ffaa00;
            font-size: 18px;
            margin: 20px;
        }
    </style>
</head>
<body>
    <h1>📷 Raspberry Pi Camera Stream</h1>
    <p class="subtitle">Live stream from /dev/video0</p>
    
    <div id="loading">Loading camera feed...</div>
    
    <div class="video-container" style="display:none;" id="video-container">
        <img id="video-feed" src="/stream.mjpg" alt="Live Camera Feed">
    </div>
    
    <div class="status" style="display:none;" id="status-div">
        Status: <span id="status">Loading...</span> | 
        Time: <span id="timestamp">--:--:--</span>
    </div>
    
    <div class="controls" style="display:none;" id="controls">
        <button onclick="reloadStream()">🔄 Reload Stream</button>
        <button onclick="location.reload()">↻ Refresh Page</button>
    </div>
    
    <script>
        const video = document.getElementById('video-feed');
        const loading = document.getElementById('loading');
        const videoContainer = document.getElementById('video-container');
        const statusDiv = document.getElementById('status-div');
        const statusSpan = document.getElementById('status');
        const timestamp = document.getElementById('timestamp');
        const controls = document.getElementById('controls');
        
        let streamLoaded = false;
        
        function updateTime() {
            const now = new Date();
            timestamp.textContent = now.toLocaleTimeString();
        }
        
        function updateStatus(text, color) {
            statusSpan.textContent = text;
            statusSpan.style.color = color || 'white';
        }
        
        function showVideo() {
            loading.style.display = 'none';
            videoContainer.style.display = 'block';
            statusDiv.style.display = 'block';
            controls.style.display = 'block';
            updateStatus('Live', '#4CAF50');
        }
        
        video.onload = function() {
            if (!streamLoaded) {
                streamLoaded = true;
                showVideo();
                console.log('Video stream loaded successfully');
            }
        };
        
        video.onerror = function() {
            updateStatus('Error', '#ff4444');
            loading.textContent = 'Failed to load stream. Retrying...';
            loading.style.color = '#ff4444';
            
            setTimeout(() => {
                video.src = '/stream.mjpg?t=' + Date.now();
            }, 2000);
        };
        
        function reloadStream() {
            updateStatus('Reloading...', '#ffaa00');
            video.src = '/stream.mjpg?t=' + Date.now();
        }
        
        // Auto-update time
        setInterval(updateTime, 1000);
        updateTime();
        
        // Start the stream
        video.src = '/stream.mjpg?t=' + Date.now();
        
        // If no stream after 10 seconds, try again
        setTimeout(() => {
            if (!streamLoaded) {
                console.log('No stream after 10 seconds, retrying...');
                video.src = '/stream.mjpg?t=' + Date.now();
            }
        }, 10000);
        
        // Auto-reload every 30 seconds
        setInterval(() => {
            if (streamLoaded) {
                console.log('Auto-refreshing stream...');
                video.src = '/stream.mjpg?t=' + Date.now();
            }
        }, 30000);
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return HTML

@app.route('/stream.mjpg')
def stream():
    def generate():
        try:
            log("🎥 New client connected")
            
            # Ensure camera is running
            if not camera_running:
                log("⚠️ Starting camera for new client")
                if not start_camera():
                    log("❌ Failed to start camera")
                    return
            
            if not camera_process or camera_process.poll() is not None:
                log("❌ Camera process not running")
                return
            
            log("📡 Streaming MJPEG data...")
            
            # Read from ffmpeg and stream to client
            while True:
                if camera_process and camera_process.stdout:
                    # Read a chunk of data
                    chunk = camera_process.stdout.read(1024 * 16)  # 16KB chunks
                    if chunk:
                        yield chunk
                    else:
                        log("⚠️ No data from camera, stream ended")
                        break
                else:
                    log("❌ Camera stdout not available")
                    break
                    
        except Exception as e:
            log(f"❌ Stream error: {e}")
        finally:
            log("🎥 Client disconnected")
    
    # Important: Use the correct content type for MJPEG stream
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Accel-Buffering': 'no'  # Disable buffering for nginx (if used)
        }
    )

@app.route('/status')
def status():
    return {
        'camera_running': camera_running,
        'camera_process_alive': camera_process.poll() is None if camera_process else False
    }

def signal_handler(sig, frame):
    log("\n🛑 Stopping camera streamer...")
    stop_camera()
    log("✅ Cleanup complete")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("🚀 STARTING WORKING CAMERA STREAMER")
    print("="*70)
    
    # Test with a simple command first
    log("Testing camera with simple capture...")
    test_cmd = ['ffmpeg', '-f', 'v4l2', '-i', '/dev/video0', '-frames:v', '1', '-f', 'image2', '-']
    try:
        result = subprocess.run(test_cmd, capture_output=True, timeout=5)
        if result.returncode == 0:
            log("✅ Camera test successful")
        else:
            log(f"❌ Camera test failed: {result.stderr.decode()[:100]}")
    except Exception as e:
        log(f"❌ Camera test error: {e}")
    
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
    print(f"\n📊 Check status: http://localhost:5000/status")
    print("\n🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Start camera
    start_camera()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)