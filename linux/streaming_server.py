#!/usr/bin/env python3
# minimal_streaming_server.py
import cv2
import threading
import time
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler

class MinimalStreamer:
    def __init__(self, port=5000):
        self.port = port
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
    
    def update_frame(self, frame):
        with self.lock:
            self.frame = frame
    
    def run(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.end_headers()
                    
                    while self.server.streamer.running:
                        with self.server.streamer.lock:
                            if self.server.streamer.frame is not None:
                                _, jpeg = cv2.imencode('.jpg', self.server.streamer.frame)
                                self.wfile.write(b'--frame\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                                self.wfile.write(jpeg.tobytes())
                                self.wfile.write(b'\r\n')
                        time.sleep(0.033)
                
                elif self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b'''
                    <html><head>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>body{margin:0;background:#000;}img{width:100%;height:100vh;object-fit:contain;}</style>
                    </head><body><img src="/stream.mjpg"></body></html>
                    ''')
        
        class Server(HTTPServer):
            def __init__(self, *args, **kwargs):
                self.streamer = kwargs.pop('streamer')
                super().__init__(*args, **kwargs)
        
        try:
            server = Server(('0.0.0.0', self.port), Handler, streamer=self)
            print(f"📹 Stream: http://{self._get_ip()}:{self.port}")
            server.serve_forever()
        except Exception as e:
            print(f"Stream error: {e}")
    
    def _get_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 1))
            return s.getsockname()[0]
        except:
            return "YOUR_PI_IP"
    
    def stop(self):
        self.running = False