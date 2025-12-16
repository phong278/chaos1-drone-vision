#!/usr/bin/env python3
"""
Simple Camera Streamer with Debug Mode
"""

import subprocess
import time
import threading
from flask import Flask, Response
import signal
import sys
import os

print("📷 SIMPLE CAMERA STREAMER")
print("="*60)

app = Flask(__name__)

# Global variables
camera_process = None
camera_running = False
debug_mode = True

def log(msg):
    """Simple logging"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def test_camera():
    """Test if camera is accessible"""
    log("🔍 Testing camera...")
    
    # Try to list camera formats
    cmd = ['v4l2-ctl', '--device=/dev/video0', '--list-formats']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            log("✅ Camera formats:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            log(f"❌ v4l2-ctl failed: {result.stderr}")
    except FileNotFoundError:
        log("⚠️ v4l2-ctl not installed. Installing...")
        subprocess.run(['sudo', 'apt', 'install', '-y', 'v4l-utils'])
        return test_camera()
    except Exception as e:
        log(f"❌ Camera test error: {e}")
    
    return False

def start_camera_stream():
    """Start camera stream with retry logic"""
    global camera_process, camera_running
    
    if camera_running and camera_process and camera_process.poll() is None:
        log("⚠️ Camera already running")
        return True
    
    # Stop any existing process
    if camera_process:
        camera_process.terminate()
        camera_process.wait()
    
    log("🚀 Starting camera stream...")
    
    # Try different camera commands
    commands_to_try = [
        # Try 1: Simple MJPEG stream
        [
            'ffmpeg',
            '-f', 'v4l2',
            '-input_format', 'mjpeg',  # Many Pi cameras use MJPEG
            '-video_size', '640x480',
            '-framerate', '15',  # Lower FPS for stability
            '-i', '/dev/video0',
            '-f', 'mjpeg',
            '-q:v', '5',
            '-r', '15',  # Output framerate
            '-'
        ],
        # Try 2: Without specifying input format
        [
            'ffmpeg',
            '-f', 'v4l2',
            '-video_size', '640x480',
            '-framerate', '10',
            '-i', '/dev/video0',
            '-f', 'mjpeg',
            '-q:v', '10',
            '-r', '10',
            '-'
        ],
        # Try 3: Very simple
        [
            'ffmpeg',
            '-f', 'v4l2',
            '-i', '/dev/video0',
            '-f', 'mjpeg',
            '-q:v', '10',
            '-'
        ]
    ]
    
    for i, cmd in enumerate(commands_to_try, 1):
        log(f"Trying command {i}/3: {' '.join(cmd[:6])}...")
        
        try:
            camera_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if debug_mode else subprocess.DEVNULL,
                bufsize=10**6
            )
            
            # Start stderr reader thread
            if debug_mode:
                def read_stderr():
                    while True:
                        line = camera_process.stderr.readline()
                        if line:
                            log(f"FFMPEG: {line.decode().strip()}")
                        elif camera_process.poll() is not None:
                            break
                        time.sleep(0.1)
                
                stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                stderr_thread.start()
            
            # Wait a bit to see if it starts successfully
            time.sleep(2)
            
            if camera_process.poll() is None:
                camera_running = True
                log(f"✅ Camera started with command {i}")
                return True
            else:
                log(f"❌ Command {i} failed")
                camera_process.terminate()
                camera_process.wait()
                
        except Exception as e:
            log(f"❌ Command {i} error: {e}")
    
    log("❌ All camera commands failed")
    camera_running = False
    return False

def stop_camera():
    """Stop camera stream"""
    global camera_process, camera_running
    
    if camera_process:
        log("📴 Stopping camera...")
        camera_process.terminate()
        try:
            camera_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            camera_process.kill()
            camera_process.wait()
        camera_process = None
    
    camera_running = False
    log("✅ Camera stopped")

# HTML page
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
        
        .header {
            margin-bottom: 20px;
        }
        
        h1 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        
        .video-container {
            max-width: 800px;
            margin: 0 auto 20px;
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
            margin: 20px 0;
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
        
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        
        .log {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #333;
            border-radius: 5px;
            padding: 10px;
            margin: 20px auto;
            max-width: 800px;
            max-height: 200px;
            overflow-y: auto;
            text-align: left;
            font-family: monospace;
            font-size: 12px;
        }
        
        .error {
            color: #ff4444;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📷 Raspberry Pi Camera Stream</h1>
        <p>Direct stream from /dev/video0</p>
    </div>
    
    <div class="video-container">
        <img id="video-feed" src="/stream.mjpg" alt="Live Camera Feed">
    </div>
    
    <div class="status">
        Status: <span id="status">Loading...</span> | 
        Last update: <span id="timestamp">--:--:--</span>
    </div>
    
    <div class="controls">
        <button onclick="reloadStream()" id="reload-btn">🔄 Reload Stream</button>
        <button onclick="restartCamera()" id="restart-btn">🔄 Restart Camera</button>
        <button onclick="location.reload()">↻ Refresh Page</button>
    </div>
    
    <div id="error" class="error"></div>
    
    <div class="log">
        <div id="log-content">Connecting to camera...</div>
    </div>
    
    <script>
        const video = document.getElementById('video-feed');
        const status = document.getElementById('status');
        const timestamp = document.getElementById('timestamp');
        const errorDiv = document.getElementById('error');
        const logContent = document.getElementById('log-content');
        const reloadBtn = document.getElementById('reload-btn');
        const restartBtn = document.getElementById('restart-btn');
        
        let connectionAttempts = 0;
        const maxAttempts = 5;
        
        function updateTime() {
            const now = new Date();
            timestamp.textContent = now.toLocaleTimeString();
        }
        
        function addLog(msg) {
            const now = new Date();
            const timeStr = now.toLocaleTimeString();
            logContent.innerHTML = `[${timeStr}] ${msg}<br>` + logContent.innerHTML;
        }
        
        function updateStatus(text, color) {
            status.textContent = text;
            status.style.color = color || 'white';
        }
        
        video.onload = function() {
            updateStatus('Live', '#4CAF50');
            errorDiv.textContent = '';
            connectionAttempts = 0;
            addLog('Video stream loaded');
        };
        
        video.onerror = function() {
            connectionAttempts++;
            
            if (connectionAttempts >= maxAttempts) {
                updateStatus('Error', '#ff4444');
                errorDiv.textContent = `Failed to load stream after ${maxAttempts} attempts. Try restarting camera.`;
                addLog('Stream error - max attempts reached');
            } else {
                updateStatus('Reconnecting...', '#ffaa00');
                errorDiv.textContent = `Connection attempt ${connectionAttempts}/${maxAttempts}`;
                addLog(`Connection attempt ${connectionAttempts} failed, retrying...`);
                
                // Exponential backoff
                const delay = Math.min(1000 * Math.pow(2, connectionAttempts - 1), 10000);
                
                setTimeout(() => {
                    video.src = '/stream.mjpg?t=' + Date.now();
                }, delay);
            }
        };
        
        function reloadStream() {
            reloadBtn.disabled = true;
            updateStatus('Reloading...', '#ffaa00');
            addLog('Manual reload requested');
            
            video.src = '/stream.mjpg?t=' + Date.now();
            
            setTimeout(() => {
                reloadBtn.disabled = false;
            }, 2000);
        }
        
        function restartCamera() {
            restartBtn.disabled = true;
            updateStatus('Restarting camera...', '#ffaa00');
            addLog('Camera restart requested');
            
            fetch('/restart')
                .then(response => response.json())
                .then(data => {
                    addLog('Camera restart: ' + data.message);
                    if (data.success) {
                        setTimeout(() => {
                            video.src = '/stream.mjpg?t=' + Date.now();
                            restartBtn.disabled = false;
                        }, 3000);
                    } else {
                        errorDiv.textContent = 'Failed to restart camera';
                        restartBtn.disabled = false;
                    }
                })
                .catch(error => {
                    addLog('Restart error: ' + error);
                    restartBtn.disabled = false;
                });
        }
        
        // Auto-refresh timestamp
        setInterval(updateTime, 1000);
        updateTime();
        
        // Initial load
        video.src = '/stream.mjpg?t=' + Date.now();
        
        // Auto-reload stream every 30 seconds
        setInterval(() => {
            if (status.textContent === 'Live') {
                addLog('Auto-reloading stream...');
                video.src = '/stream.mjpg?t=' + Date.now();
            }
        }, 30000);
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
            if not camera_running or camera_process is None or camera_process.poll() is not None:
                log("⚠️ Camera not running, attempting to start...")
                start_camera_stream()
                time.sleep(2)  # Give it time to start
            
            if camera_running and camera_process and camera_process.poll() is None:
                log("🎥 Starting stream to client")
                
                # Stream directly from ffmpeg
                while True:
                    if camera_process and camera_process.stdout:
                        chunk = camera_process.stdout.read(1024)
                        if chunk:
                            yield chunk
                        else:
                            log("⚠️ No data from camera")
                            break
                    else:
                        log("⚠️ Camera process not available")
                        break
                
                log("🎥 Stream ended")
            else:
                log("❌ Cannot stream: camera not available")
                
                # Send error image
                error_img = create_error_image("Camera Not Available")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       error_img + b'\r\n')
                
        except Exception as e:
            log(f"❌ Stream error: {e}")
            error_img = create_error_image(f"Error: {str(e)[:50]}")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   error_img + b'\r\n')
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={
                       'Cache-Control': 'no-cache, no-store, must-revalidate',
                       'Pragma': 'no-cache',
                       'Expires': '0'
                   })

@app.route('/restart')
def restart():
    """Restart camera endpoint"""
    log("🔄 Manual restart requested")
    stop_camera()
    success = start_camera_stream()
    
    return {
        'success': success,
        'message': 'Camera restarted' if success else 'Failed to restart camera'
    }

def create_error_image(message):
    """Create a simple error image"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        img = Image.new('RGB', (640, 480), color='black')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw error message
        draw.text((50, 200), message, fill='red', font=font)
        draw.text((50, 250), "Check /dev/video0 connection", fill='white', font=font)
        
        # Save to bytes
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        return buf.getvalue()
        
    except ImportError:
        # If PIL is not available, return a simple colored image
        import struct
        # Create a simple red gradient
        width, height = 640, 480
        data = b''
        for y in range(height):
            for x in range(width):
                r = 255 if y < height//2 else 128
                g = 0
                b = 0
                data += struct.pack('BBB', r, g, b)
        return data

def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    log("\n🛑 Stopping camera streamer...")
    stop_camera()
    log("✅ Cleanup complete")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("🚀 SIMPLE CAMERA STREAMER")
    print("="*70)
    
    # Check dependencies
    log("Checking dependencies...")
    dependencies = ['ffmpeg']
    missing_deps = []
    
    for dep in dependencies:
        try:
            subprocess.run([dep, '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"✅ {dep} is installed")
        except FileNotFoundError:
            log(f"❌ {dep} not found")
            missing_deps.append(dep)
    
    if missing_deps:
        log(f"\n⚠️  Missing: {', '.join(missing_deps)}")
        log("Install with: sudo apt update && sudo apt install ffmpeg")
        response = input("Install now? (y/n): ")
        if response.lower() == 'y':
            subprocess.run(['sudo', 'apt', 'update'])
            subprocess.run(['sudo', 'apt', 'install', '-y'] + missing_deps)
    
    # Test camera
    if not test_camera():
        log("⚠️  Camera test failed, but continuing anyway...")
    
    # Start camera
    start_camera_stream()
    
    # Get IP address
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
    except:
        local_ip = '127.0.0.1'
    
    print(f"\n🌐 Stream available at:")
    print(f"   • http://localhost:5000")
    print(f"   • http://{local_ip}:5000")
    print("\n🔧 Debug info:")
    print(f"   • Camera running: {camera_running}")
    print(f"   • Debug mode: {debug_mode}")
    print("\n🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)