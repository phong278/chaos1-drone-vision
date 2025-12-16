#!/usr/bin/env python3
"""
Fixed Camera Streamer with correct routes
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

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def start_camera():
    global camera_process, camera_running
    
    if camera_running and camera_process and camera_process.poll() is None:
        return True
    
    # Stop existing process
    if camera_process:
        camera_process.terminate()
        camera_process.wait()
    
    log("🚀 Starting camera...")
    
    # Use the command that worked from your output
    cmd = [
        'ffmpeg',
        '-f', 'v4l2',
        '-input_format', 'mjpeg',
        '-video_size', '640x480',
        '-framerate', '15',
        '-i', '/dev/video0',
        '-f', 'mjpeg',
        '-q:v', '5',
        '-r', '15',
        '-'
    ]
    
    try:
        camera_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**6
        )
        
        # Read stderr in background
        def read_stderr():
            while True:
                line = camera_process.stderr.readline()
                if line:
                    line_str = line.decode().strip()
                    if not line_str.startswith('frame='):  # Filter out frame stats
                        log(f"FFMPEG: {line_str}")
                elif camera_process.poll() is not None:
                    break
                time.sleep(0.1)
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        # Wait to see if it starts
        time.sleep(2)
        
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
        camera_process.terminate()
        try:
            camera_process.wait(timeout=3)
        except:
            camera_process.kill()
        camera_process = None
    camera_running = False

# HTML with FIXED route
HTML = '''
<!DOCTYPE html>
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
            margin-bottom: 10px;
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
        
        .error {
            color: #ff4444;
            margin: 10px;
            padding: 10px;
            background: rgba(255, 68, 68, 0.1);
            border-radius: 5px;
        }
        
        .success {
            color: #4CAF50;
            margin: 10px;
            padding: 10px;
            background: rgba(76, 175, 80, 0.1);
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>📷 Raspberry Pi Camera Stream</h1>
    <p>Live stream from /dev/video0</p>
    
    <div class="video-container">
        <img id="video-feed" src="/stream.mjpg" alt="Live Camera Feed">
    </div>
    
    <div class="status">
        Status: <span id="status">Loading...</span> | 
        Time: <span id="timestamp">--:--:--</span>
    </div>
    
    <div class="controls">
        <button onclick="reloadStream()">🔄 Reload Stream</button>
        <button onclick="location.reload()">↻ Refresh Page</button>
    </div>
    
    <div id="message"></div>
    
    <script>
        const video = document.getElementById('video-feed');
        const status = document.getElementById('status');
        const timestamp = document.getElementById('timestamp');
        const messageDiv = document.getElementById('message');
        
        let reloadTimeout;
        
        function updateTime() {
            const now = new Date();
            timestamp.textContent = now.toLocaleTimeString();
        }
        
        function showMessage(text, type) {
            messageDiv.textContent = text;
            messageDiv.className = type;
            messageDiv.style.display = 'block';
            
            clearTimeout(reloadTimeout);
            if (type === 'error') {
                reloadTimeout = setTimeout(() => {
                    showMessage('Auto-reloading...', 'success');
                    reloadStream();
                }, 3000);
            }
        }
        
        function updateStatus(text, color) {
            status.textContent = text;
            status.style.color = color || 'white';
        }
        
        video.onload = function() {
            updateStatus('Live', '#4CAF50');
            showMessage('Stream connected successfully!', 'success');
        };
        
        video.onerror = function() {
            updateStatus('Error', '#ff4444');
            showMessage('Failed to load stream. Retrying...', 'error');
            
            // Retry after 2 seconds
            setTimeout(() => {
                video.src = '/stream.mjpg?t=' + Date.now();
            }, 2000);
        };
        
        function reloadStream() {
            updateStatus('Reloading...', '#ffaa00');
            showMessage('Reloading stream...', 'success');
            video.src = '/stream.mjpg?t=' + Date.now();
        }
        
        // Auto-update time
        setInterval(updateTime, 1000);
        updateTime();
        
        // Initial load
        video.src = '/stream.mjpg?t=' + Date.now();
        
        // Auto-reload every 60 seconds to prevent freezing
        setInterval(() => {
            if (status.textContent === 'Live') {
                reloadStream();
            }
        }, 60000);
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return HTML

@app.route('/stream.mjpg')
def stream():
    def generate():
        try:
            # Ensure camera is running
            if not camera_running:
                log("⚠️ Camera not running, starting...")
                if not start_camera():
                    log("❌ Failed to start camera")
                    return
            
            if camera_process and camera_process.poll() is None:
                log("🎥 Streaming to client...")
                
                # Stream directly from ffmpeg
                while True:
                    if camera_process.stdout:
                        chunk = camera_process.stdout.read(1024)
                        if chunk:
                            yield chunk
                        else:
                            log("⚠️ No data from camera")
                            break
                    else:
                        log("⚠️ Camera stdout not available")
                        break
                        
            else:
                log("❌ Camera process not available")
                
        except Exception as e:
            log(f"❌ Stream error: {e}")
        finally:
            log("🎥 Stream ended")
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={
                       'Cache-Control': 'no-cache, no-store, must-revalidate',
                       'Pragma': 'no-cache',
                       'Expires': '0'
                   })

def signal_handler(sig, frame):
    log("\n🛑 Stopping camera streamer...")
    stop_camera()
    log("✅ Cleanup complete")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("🚀 STARTING CAMERA STREAMER")
    print("="*70)
    
    # Test camera
    log("Testing camera...")
    test_cmd = ['v4l2-ctl', '--device=/dev/video0', '--list-formats']
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Camera supports:")
            for line in result.stdout.split('\n'):
                if 'MJPG' in line or 'YUYV' in line:
                    print(f"  {line.strip()}")
        else:
            print("Could not query camera")
    except:
        print("v4l2-ctl not available")
    
    # Start camera
    if start_camera():
        print(f"\n✅ Camera is running")
    else:
        print(f"\n❌ Failed to start camera")
        print("Try manually: ffplay -f v4l2 -input_format mjpeg -video_size 640x480 /dev/video0")
    
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
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)