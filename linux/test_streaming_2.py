#!/usr/bin/env python3
"""
Ultra Simple Pi Camera Streamer
Directly streams from ffmpeg to browser - no processing
"""

import subprocess
import time
from flask import Flask, Response, render_template_string
import signal
import sys
import os

print("📷 ULTRA SIMPLE PI CAMERA STREAMER")
print("="*60)

app = Flask(__name__)
camera_process = None

def start_camera():
    """Start ffmpeg to stream camera directly as MJPEG"""
    global camera_process
    
    try:
        print("🚀 Starting camera stream...")
        
        # Simple ffmpeg command that outputs MJPEG stream
        cmd = [
            'ffmpeg',
            '-f', 'v4l2',
            '-input_format', 'mjpeg',  # Many Pi cameras output MJPEG natively
            '-video_size', '640x480',
            '-framerate', '30',
            '-i', '/dev/video0',
            '-f', 'mjpeg',
            '-q:v', '5',  # Quality: 2-31 (lower is better)
            '-'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        camera_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**6
        )
        
        # Read stderr in background to see errors
        def read_stderr():
            while True:
                line = camera_process.stderr.readline()
                if line:
                    print(f"FFMPEG: {line.decode().strip()}")
                else:
                    break
        
        import threading
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        print("✅ Camera process started")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start camera: {e}")
        return False

def stop_camera():
    """Stop the camera process"""
    global camera_process
    if camera_process:
        print("📴 Stopping camera...")
        camera_process.terminate()
        camera_process.wait()
        camera_process = None

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📷 Pi Camera Live</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            background: #0a0a0a;
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
        }
        
        h1 {
            font-size: 2.5rem;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #aaa;
            font-size: 1.1rem;
        }
        
        .video-container {
            max-width: 800px;
            width: 100%;
            background: black;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #333;
            margin-bottom: 20px;
        }
        
        #video-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .status {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
            font-family: monospace;
        }
        
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }
        
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
        }
        
        button:hover {
            background: #45a049;
        }
        
        .error {
            color: #ff4444;
            margin-top: 10px;
            padding: 10px;
            background: rgba(255, 68, 68, 0.1);
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📷 Raspberry Pi Camera</h1>
        <p class="subtitle">Live stream from /dev/video0</p>
    </div>
    
    <div class="video-container">
        <img id="video-feed" src="/video.mjpg" alt="Live Camera Feed">
    </div>
    
    <div class="status">
        <div>Stream: <span id="status">Loading...</span></div>
        <div>Last update: <span id="time">--:--:--</span></div>
    </div>
    
    <div class="controls">
        <button onclick="reloadStream()">🔄 Reload Stream</button>
        <button onclick="takeSnapshot()">📸 Take Snapshot</button>
    </div>
    
    <div id="error" class="error" style="display: none;"></div>
    
    <script>
        const videoFeed = document.getElementById('video-feed');
        const status = document.getElementById('status');
        const timeDisplay = document.getElementById('time');
        const errorDiv = document.getElementById('error');
        
        // Update time display
        function updateTime() {
            const now = new Date();
            timeDisplay.textContent = now.toLocaleTimeString();
        }
        
        setInterval(updateTime, 1000);
        updateTime();
        
        // Handle video load events
        videoFeed.onload = function() {
            status.textContent = 'Live';
            status.style.color = '#4CAF50';
            errorDiv.style.display = 'none';
        };
        
        videoFeed.onerror = function() {
            status.textContent = 'Error';
            status.style.color = '#ff4444';
            errorDiv.textContent = 'Failed to load stream. Trying again...';
            errorDiv.style.display = 'block';
            
            // Try to reload after 2 seconds
            setTimeout(() => {
                videoFeed.src = '/video.mjpg?t=' + Date.now();
            }, 2000);
        };
        
        // Manual reload
        function reloadStream() {
            status.textContent = 'Reconnecting...';
            status.style.color = '#ffaa00';
            videoFeed.src = '/video.mjpg?t=' + Date.now();
        }
        
        // Take snapshot
        function takeSnapshot() {
            const canvas = document.createElement('canvas');
            canvas.width = videoFeed.videoWidth || 640;
            canvas.height = videoFeed.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoFeed, 0, 0);
            
            const link = document.createElement('a');
            link.download = 'snapshot_' + new Date().toISOString().replace(/[:.]/g, '-') + '.jpg';
            link.href = canvas.toDataURL('image/jpeg');
            link.click();
            
            alert('Snapshot saved!');
        }
        
        // Auto-reload every 30 seconds to prevent freezing
        setInterval(reloadStream, 30000);
        
        // Initial load
        videoFeed.src = '/video.mjpg?t=' + Date.now();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video.mjpg')
def video_feed():
    """Stream camera directly from ffmpeg"""
    def generate():
        try:
            # Check if camera process is running
            if not camera_process or camera_process.poll() is not None:
                print("⚠️ Camera process not running, restarting...")
                start_camera()
                time.sleep(1)  # Give it time to start
            
            # Stream ffmpeg output directly to browser
            while True:
                if camera_process and camera_process.stdout:
                    # Read chunk from ffmpeg
                    chunk = camera_process.stdout.read(1024)
                    if chunk:
                        yield chunk
                    else:
                        # No data, try to restart
                        print("⚠️ No data from camera, restarting...")
                        stop_camera()
                        start_camera()
                        time.sleep(2)
                        break
                else:
                    print("⚠️ Camera process not available")
                    time.sleep(1)
                    break
                    
        except Exception as e:
            print(f"❌ Stream error: {e}")
            # Return an error image
            error_img = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + get_error_image() + b'\r\n'
            yield error_img
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={
                       'Cache-Control': 'no-cache, no-store, must-revalidate',
                       'Pragma': 'no-cache',
                       'Expires': '0'
                   })

def get_error_image():
    """Create a simple error image"""
    from PIL import Image, ImageDraw
    import io
    
    img = Image.new('RGB', (640, 480), color='black')
    draw = ImageDraw.Draw(img)
    
    # Simple text without font
    draw.text((200, 200), "CAMERA ERROR", fill='white')
    draw.text((180, 230), "Check /dev/video0", fill='red')
    
    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=80)
    return buf.getvalue()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Stopping camera streamer...")
    stop_camera()
    print("✅ Cleanup complete")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("🚀 STARTING ULTRA SIMPLE CAMERA STREAMER")
    print("="*70)
    
    # First, let's test the camera directly
    print("\n🔍 Testing camera with simple command...")
    test_cmd = ['ffmpeg', '-f', 'v4l2', '-list_formats', 'all', '-i', '/dev/video0']
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Camera device found")
            print("Supported formats:")
            for line in result.stderr.split('\n'):
                if 'v4l2' in line or 'mjpeg' in line or 'yuyv' in line:
                    print(f"  {line.strip()}")
        else:
            print("❌ Cannot access camera device")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing camera: {e}")
    
    # Try alternative camera device
    print("\n🔍 Looking for camera devices...")
    import glob
    video_devices = glob.glob('/dev/video*')
    print(f"Found devices: {video_devices}")
    
    # Try to start camera
    camera_started = start_camera()
    
    if not camera_started:
        print("\n⚠️  Could not start camera. Using fallback...")
    
    # Get IP address
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))
        local_ip = s.getsockname()[0]
    except:
        local_ip = '127.0.0.1'
    
    print(f"\n🌐 Stream available at:")
    print(f"   • http://localhost:5000")
    print(f"   • http://{local_ip}:5000")
    print("\n📹 If you see 'CAMERA ERROR', check:")
    print("   1. Camera is connected: ls /dev/video*")
    print("   2. Permissions: sudo chmod 666 /dev/video0")
    print("   3. Try: ffplay -f v4l2 /dev/video0")
    print("🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)