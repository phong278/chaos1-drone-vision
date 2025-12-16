#!/usr/bin/env python3
"""
Simple Pi Camera Streamer - No OpenCV Required
Streams Raspberry Pi camera to a clean webpage using ffmpeg and PIL.
"""

import time
import subprocess
import threading
from flask import Flask, Response, render_template_string
import signal
import sys
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

print("📷 SIMPLE PI CAMERA STREAMER (No OpenCV)")
print("="*60)

class PiCameraStreamer:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_process = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.frame_count = 0
        
        # Try to load a font, fall back to default
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            self.font = ImageFont.load_default()
        
        print(f"📹 Camera: {width}x{height} @ {fps}fps")
    
    def start_camera(self):
        """Start the camera using ffmpeg command"""
        try:
            # FFMPEG command to capture from camera and output JPEG
            cmd = [
                'ffmpeg',
                '-f', 'v4l2',
                '-framerate', str(self.fps),
                '-video_size', f'{self.width}x{self.height}',
                '-i', '/dev/video0',
                '-f', 'image2pipe',
                '-vf', 'format=yuv420p',
                '-vcodec', 'mjpeg',
                '-q:v', '2',  # Quality factor (2-31, lower is better)
                '-'
            ]
            
            print(f"🚀 Starting camera: {' '.join(cmd[:10])}...")
            self.camera_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**6  # 1MB buffer
            )
            
            self.running = True
            self.read_thread = threading.Thread(target=self.read_frames, daemon=True)
            self.read_thread.start()
            
            print("✅ Camera started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start camera: {e}")
            print("\n🔧 Troubleshooting:")
            print("   1. Check camera is connected: ls /dev/video*")
            print("   2. Try: ffplay -f v4l2 -video_size 640x480 /dev/video0")
            print("   3. Install ffmpeg: sudo apt install ffmpeg")
            print("   4. Install PIL: pip3 install Pillow")
            
            # Create a test pattern if camera fails
            self.create_test_pattern()
    
    def create_test_pattern(self):
        """Create a test pattern if camera fails"""
        print("⚠️  Creating test pattern (camera not available)")
        self.running = True
        self.test_thread = threading.Thread(target=self.generate_test_pattern, daemon=True)
        self.test_thread.start()
    
    def generate_test_pattern(self):
        """Generate a moving test pattern using PIL"""
        while self.running:
            # Create a new image
            img = Image.new('RGB', (self.width, self.height), color='black')
            draw = ImageDraw.Draw(img)
            
            # Add moving elements
            t = time.time()
            center_x = self.width // 2
            center_y = self.height // 2
            radius = min(self.width, self.height) // 4
            
            # Moving circle
            x = int(center_x + radius * np.sin(t))
            y = int(center_y + radius * np.cos(t * 0.7))
            
            # Draw circle
            draw.ellipse([x-30, y-30, x+30, y+30], fill=(0, 200, 255))
            
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            draw.text((10, 10), f"TEST MODE - {timestamp}", fill=(255, 255, 255), font=self.font)
            draw.text((10, 40), "Camera not detected", fill=(200, 200, 200), font=self.font)
            draw.text((10, 70), "Check /dev/video0 connection", fill=(200, 200, 200), font=self.font)
            
            # Add frame counter
            self.frame_count += 1
            draw.text((self.width - 150, 10), f"Frame: {self.frame_count}", fill=(150, 150, 150), font=self.font)
            
            # Convert to bytes
            with BytesIO() as output:
                img.save(output, format='JPEG', quality=85)
                jpeg_data = output.getvalue()
            
            with self.frame_lock:
                self.latest_frame = jpeg_data
            
            time.sleep(1/self.fps)
    
    def read_frames(self):
        """Read JPEG frames from ffmpeg process"""
        print(f"📥 Reading JPEG frames from camera...")
        
        # Read JPEG frames from ffmpeg output
        while self.running and self.camera_process:
            try:
                # FFmpeg outputs JPEG frames with markers
                # Look for JPEG start marker (0xFF 0xD8)
                while True:
                    # Read until we find start of JPEG
                    byte = self.camera_process.stdout.read(1)
                    if not byte:
                        break
                    
                    if ord(byte) == 0xFF:
                        next_byte = self.camera_process.stdout.read(1)
                        if next_byte and ord(next_byte) == 0xD8:
                            # Found JPEG start, now read the rest
                            jpeg_data = byte + next_byte
                            
                            # Read until we find JPEG end marker (0xFF 0xD9)
                            while True:
                                chunk = self.camera_process.stdout.read(1024)
                                if not chunk:
                                    break
                                jpeg_data += chunk
                                
                                # Check if we have the end marker
                                if len(jpeg_data) >= 2 and jpeg_data[-2:] == b'\xFF\xD9':
                                    # Complete JPEG frame
                                    
                                    # Add timestamp overlay using PIL
                                    img = Image.open(BytesIO(jpeg_data))
                                    draw = ImageDraw.Draw(img)
                                    
                                    # Add timestamp
                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                    draw.text((10, 10), timestamp, fill=(255, 255, 255), font=self.font)
                                    
                                    # Add resolution info
                                    info_text = f"{self.width}x{self.height} @ {self.fps}fps"
                                    draw.text((10, 40), info_text, fill=(200, 200, 255), font=self.font)
                                    
                                    # Save back to JPEG
                                    with BytesIO() as output:
                                        img.save(output, format='JPEG', quality=85)
                                        jpeg_data = output.getvalue()
                                    
                                    with self.frame_lock:
                                        self.latest_frame = jpeg_data
                                    
                                    self.frame_count += 1
                                    break
                            
                            break  # Exit the byte reading loop to get next frame
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                    
            except Exception as e:
                print(f"⚠️  Error reading frame: {e}")
                break
        
        print("📴 Camera feed ended")
    
    def get_frame(self):
        """Get the latest frame as JPEG bytes"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame
        
        # Return a placeholder frame
        img = Image.new('RGB', (self.width, self.height), color='black')
        draw = ImageDraw.Draw(img)
        draw.text((self.width//2 - 150, self.height//2 - 20), 
                 "NO CAMERA FEED", fill=(255, 255, 255), font=self.font)
        
        with BytesIO() as output:
            img.save(output, format='JPEG', quality=85)
            return output.getvalue()
    
    def stop(self):
        """Stop the camera"""
        self.running = False
        if self.camera_process:
            self.camera_process.terminate()
            self.camera_process.wait()
        print("📴 Camera stopped")

# Create Flask app
app = Flask(__name__)

# Initialize camera
camera = PiCameraStreamer(width=640, height=480, fps=30)

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📷 Raspberry Pi Camera</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            color: #f0f0f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            width: 100%;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            animation: fadeIn 0.8s ease-out;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            color: #aaa;
            font-size: 1.1rem;
            margin-bottom: 20px;
        }
        
        .video-container {
            background: rgba(0, 0, 0, 0.5);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            animation: fadeIn 1s ease-out;
            display: flex;
            justify-content: center;
        }
        
        .video-wrapper {
            position: relative;
            border-radius: 10px;
            overflow: hidden;
            max-width: 800px;
            width: 100%;
            aspect-ratio: 4/3;
        }
        
        #video-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        
        .loading {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.5s ease-out;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            animation: fadeIn 1.2s ease-out;
        }
        
        .status-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px 25px;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            min-width: 200px;
            text-align: center;
            transition: transform 0.3s;
        }
        
        .status-item:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
        }
        
        .status-item i {
            font-size: 1.5rem;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        
        .status-label {
            font-size: 0.9rem;
            color: #aaa;
            margin-bottom: 5px;
        }
        
        .status-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        footer {
            text-align: center;
            color: #666;
            padding: 20px;
            margin-top: auto;
            animation: fadeIn 1.4s ease-out;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            h1 {
                font-size: 2rem;
            }
            
            .status-bar {
                flex-direction: column;
                align-items: center;
            }
            
            .status-item {
                width: 100%;
                max-width: 300px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><i class="fas fa-camera"></i> Raspberry Pi Camera</h1>
            <p class="subtitle">Live streaming from Raspberry Pi Camera Module</p>
        </header>
        
        <div class="video-container">
            <div class="video-wrapper">
                <img id="video-feed" src="/video_feed" alt="Live Camera Feed">
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Connecting to camera...</p>
                </div>
            </div>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <i class="fas fa-video"></i>
                <div class="status-label">Resolution</div>
                <div class="status-value" id="resolution">640x480</div>
            </div>
            <div class="status-item">
                <i class="fas fa-tachometer-alt"></i>
                <div class="status-label">Frame Rate</div>
                <div class="status-value" id="fps">30 fps</div>
            </div>
            <div class="status-item">
                <i class="fas fa-clock"></i>
                <div class="status-label">Uptime</div>
                <div class="status-value" id="uptime">0:00</div>
            </div>
        </div>
        
        <footer>
            <p><i class="fas fa-microchip"></i> Raspberry Pi Camera Streamer</p>
            <p>Streaming from /dev/video0 | No OpenCV Required</p>
        </footer>
    </div>
    
    <script>
        // Hide loading screen when video loads
        const videoFeed = document.getElementById('video-feed');
        const loading = document.getElementById('loading');
        
        videoFeed.onload = function() {
            loading.style.display = 'none';
        };
        
        videoFeed.onerror = function() {
            // If error, reload the feed
            loading.style.display = 'flex';
            setTimeout(() => {
                this.src = '/video_feed?' + new Date().getTime();
            }, 1000);
        };
        
        // Update uptime counter
        let startTime = Date.now();
        function updateUptime() {
            const elapsed = Date.now() - startTime;
            const seconds = Math.floor(elapsed / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            
            let uptimeText = '';
            if (hours > 0) {
                uptimeText = hours + ':' + (minutes % 60).toString().padStart(2, '0') + ':' + (seconds % 60).toString().padStart(2, '0');
            } else {
                uptimeText = minutes + ':' + (seconds % 60).toString().padStart(2, '0');
            }
            
            document.getElementById('uptime').textContent = uptimeText;
        }
        
        // Update uptime every second
        setInterval(updateUptime, 1000);
        updateUptime(); // Initial update
        
        // Handle page visibility
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden) {
                // Page is visible again, reload video feed
                loading.style.display = 'flex';
                videoFeed.src = '/video_feed?' + new Date().getTime();
            }
        });
        
        // Auto-reload feed every 3 seconds to prevent freezing
        setInterval(function() {
            videoFeed.src = '/video_feed?' + new Date().getTime();
        }, 3000);
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route - serves JPEG frames"""
    def generate():
        last_time = time.time()
        frame_interval = 1/30  # Target 30 FPS
        
        while True:
            try:
                # Get current time
                current_time = time.time()
                
                # Control frame rate
                elapsed = current_time - last_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                
                # Get JPEG frame from camera
                jpeg_data = camera.get_frame()
                
                # Send the frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       jpeg_data + b'\r\n')
                
                last_time = time.time()
                
            except Exception as e:
                print(f"⚠️  Stream error: {e}")
                # Create error image
                img = Image.new('RGB', (640, 480), color='black')
                draw = ImageDraw.Draw(img)
                try:
                    draw.text((200, 200), "STREAM ERROR", fill=(255, 255, 255), font=camera.font)
                except:
                    pass
                
                with BytesIO() as output:
                    img.save(output, format='JPEG', quality=85)
                    error_frame = output.getvalue()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       error_frame + b'\r\n')
                
                time.sleep(1)  # Wait before retrying
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={
                       'Cache-Control': 'no-cache, no-store, must-revalidate',
                       'Pragma': 'no-cache',
                       'Expires': '0'
                   })

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Stopping camera streamer...")
    camera.stop()
    print("✅ Cleanup complete")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("🚀 STARTING PI CAMERA STREAMER (No OpenCV)")
    print("="*70)
    
    # Install instructions
    print("\n📦 Dependencies needed:")
    print("   1. ffmpeg: sudo apt install ffmpeg")
    print("   2. Pillow: pip3 install Pillow")
    print("   3. Flask: pip3 install flask")
    print("   4. numpy: pip3 install numpy")
    
    # Check dependencies
    try:
        from PIL import Image
        print("✅ Pillow is installed")
    except ImportError:
        print("❌ Pillow not installed. Run: pip3 install Pillow")
        sys.exit(1)
    
    try:
        import flask
        print("✅ Flask is installed")
    except ImportError:
        print("❌ Flask not installed. Run: pip3 install flask")
        sys.exit(1)
    
    # Start camera
    camera.start_camera()
    
    # Give camera time to start
    time.sleep(2)
    
    # Get local IP address
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
    print("\n📹 Camera should appear in 3-5 seconds")
    print("🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)