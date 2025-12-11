#!/usr/bin/env python3
"""
Test streaming server locally - No Pi needed!
Simulates full Pi 4 detection system on your laptop.
"""

import cv2
import numpy as np
import time
import threading
import queue
import json
from flask import Flask, Response, jsonify
import socket

print("🧪 LOCAL STREAMING TEST - No Pi Required")
print("="*60)

# ==================== MOCK DETECTION SYSTEM ====================
class MockDetectionSystem:
    """Simulates Pi 4 detection without needing Pi hardware"""
    
    def __init__(self):
        self.frame_count = 0
        self.fps = 25.0
        self.temperature = 45.0
        self.detections = []
        
        # Create test patterns
        self.test_patterns = self._create_test_patterns()
        self.pattern_index = 0
        
        print("✅ Mock detection system created")
        print("   • Simulated camera feed")
        print("   • Mock object detection")
        print("   • Simulated temperature")
    
    def _create_test_patterns(self):
        """Create rotating test patterns"""
        patterns = []
        
        # Pattern 1: Person on left
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (200, 300), (0, 100, 255), -1)  # Orange person
        cv2.putText(img, "PERSON", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        patterns.append(img)
        
        # Pattern 2: Car on right
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (400, 200), (550, 300), (255, 100, 0), -1)  # Blue car
        cv2.putText(img, "CAR", (400, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        patterns.append(img)
        
        # Pattern 3: Multiple objects
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (150, 150), (250, 350), (0, 100, 255), -1)  # Person
        cv2.rectangle(img, (350, 200), (500, 280), (255, 100, 0), -1)  # Car
        cv2.rectangle(img, (300, 350), (350, 450), (100, 255, 0), -1)  # Chair
        cv2.rectangle(img, (50, 50), (120, 120), (255, 255, 100), -1)  # Bottle
        cv2.putText(img, "MULTIPLE OBJECTS", (200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        patterns.append(img)
        
        # Pattern 4: Traffic scene
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 150), (180, 320), (0, 100, 255), -1)  # Person 1
        cv2.rectangle(img, (200, 150), (280, 320), (0, 100, 255), -1)  # Person 2
        cv2.rectangle(img, (350, 200), (450, 280), (255, 100, 0), -1)  # Car 1
        cv2.rectangle(img, (480, 200), (580, 280), (255, 100, 0), -1)  # Car 2
        cv2.rectangle(img, (150, 400), (220, 450), (100, 255, 0), -1)  # Chair
        cv2.putText(img, "TRAFFIC SCENE", (250, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        patterns.append(img)
        
        # Pattern 5: Empty scene
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add gradient background
        for i in range(img.shape[0]):
            color = int(30 + (i / img.shape[0]) * 40)
            img[i, :] = (color, color, color)
        cv2.putText(img, "CLEAR - NO OBJECTS", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        patterns.append(img)
        
        return patterns
    
    def get_next_frame(self):
        """Get next test frame"""
        frame = self.test_patterns[self.pattern_index].copy()
        self.pattern_index = (self.pattern_index + 1) % len(self.test_patterns)
        self.frame_count += 1
        
        # Add frame counter
        cv2.putText(frame, f"Test Frame: {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add simulated FPS
        cv2.putText(frame, f"Simulated FPS: {self.fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add temperature
        temp_color = (0, 255, 0) if self.temperature < 70 else (0, 255, 255) if self.temperature < 80 else (0, 0, 255)
        cv2.putText(frame, f"Sim Temp: {self.temperature:.1f}°C", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, temp_color, 2)
        
        return frame
    
    def simulate_detection(self, frame):
        """Simulate object detection"""
        self.detections = []
        
        # Simulate detections based on pattern
        if self.pattern_index == 0:  # Person pattern
            self.detections = [{
                'label': 'person',
                'confidence': 0.87,
                'bbox': [100, 100, 200, 300]
            }]
        elif self.pattern_index == 1:  # Car pattern
            self.detections = [{
                'label': 'car',
                'confidence': 0.76,
                'bbox': [400, 200, 550, 300]
            }]
        elif self.pattern_index == 2:  # Multiple objects
            self.detections = [
                {'label': 'person', 'confidence': 0.87, 'bbox': [150, 150, 250, 350]},
                {'label': 'car', 'confidence': 0.76, 'bbox': [350, 200, 500, 280]},
                {'label': 'chair', 'confidence': 0.68, 'bbox': [300, 350, 350, 450]},
                {'label': 'bottle', 'confidence': 0.54, 'bbox': [50, 50, 120, 120]}
            ]
        elif self.pattern_index == 3:  # Traffic scene
            self.detections = [
                {'label': 'person', 'confidence': 0.82, 'bbox': [100, 150, 180, 320]},
                {'label': 'person', 'confidence': 0.79, 'bbox': [200, 150, 280, 320]},
                {'label': 'car', 'confidence': 0.88, 'bbox': [350, 200, 450, 280]},
                {'label': 'car', 'confidence': 0.85, 'bbox': [480, 200, 580, 280]},
                {'label': 'chair', 'confidence': 0.62, 'bbox': [150, 400, 220, 450]}
            ]
        # Pattern 4: No detections
        
        return self.detections

# ==================== OPTIMIZED STREAMING SERVER ====================
class LocalStreamingServer:
    """Full streaming server with optimized UI - identical to Pi version"""
    
    def __init__(self, port=5000, max_clients=20):
        self.port = port
        self.max_clients = max_clients
        self.active_clients = 0
        self.client_lock = threading.Lock()
        
        # Frame management
        self.latest_frame = None
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()
        
        # Performance tracking
        self.detection_info = {
            'fps': 25.0,
            'frame_count': 0,
            'detections': [],
            'temperature': 45.0,
            'processing_time': 32.5,
            'active_clients': 0,
            'server_uptime': time.time(),
            'total_frames_served': 0
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
                <title>🧪 Pi 4 Drone System - LOCAL TEST</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    :root {
                        --primary: #ff9900;
                        --secondary: #ff3300;
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
                        background: radial-gradient(circle, rgba(255, 153, 0, 0.1) 0%, transparent 70%);
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
                    
                    /* Warning Banner */
                    .warning-banner {
                        background: linear-gradient(90deg, #ff9900, #ff3300);
                        color: black;
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                        text-align: center;
                        font-weight: bold;
                        border: 2px solid #ffcc00;
                        animation: pulse-warning 2s infinite;
                    }
                    
                    @keyframes pulse-warning {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.8; }
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
                        box-shadow: 0 5px 15px rgba(255, 153, 0, 0.2);
                    }
                    
                    .status-card.simulation {
                        border-left-color: #ff9900;
                    }
                    
                    .status-card i {
                        font-size: 1.5rem;
                        margin-right: 10px;
                        color: var(--primary);
                    }
                    
                    .status-card.simulation i { color: #ff9900; }
                    
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
                    
                    /* Pattern Info */
                    .pattern-info {
                        background: var(--dark-gray);
                        padding: 15px;
                        border-radius: 10px;
                        margin: 10px 0;
                        border-left: 4px solid #00aaff;
                    }
                    
                    .pattern-list {
                        margin-top: 10px;
                        padding-left: 20px;
                    }
                    
                    .pattern-list li {
                        margin: 5px 0;
                        color: var(--text-secondary);
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <!-- Header -->
                    <header class="header fade-in">
                        <h1><i class="fas fa-flask"></i> Pi 4 Drone System - LOCAL TEST</h1>
                        <p>Full simulation of Raspberry Pi 4 detection on your laptop</p>
                    </header>
                    
                    <!-- Warning Banner -->
                    <div class="warning-banner fade-in">
                        <i class="fas fa-exclamation-triangle"></i>
                        <strong>TEST MODE ACTIVE</strong> - This is a complete simulation
                        <br>
                        No Pi hardware required. Testing all streaming & detection functionality.
                    </div>
                    
                    <!-- Connection Info -->
                    <div class="connection-info fade-in">
                        <p><i class="fas fa-wifi"></i> Local Test Server Running</p>
                        <p class="ip-address" id="ip-address">Loading server IP...</p>
                        <p><i class="fas fa-users"></i> Connected Clients: <span id="client-count">0</span>/<span id="max-clients">20</span></p>
                    </div>
                    
                    <!-- Status Bar -->
                    <div class="status-bar fade-in" style="animation-delay: 0.2s">
                        <div class="status-card simulation">
                            <h3><i class="fas fa-tachometer-alt"></i> Frame Rate</h3>
                            <div class="value" id="fps-value">25.0</div>
                            <span class="unit">FPS (simulated)</span>
                        </div>
                        <div class="status-card simulation">
                            <h3><i class="fas fa-thermometer-half"></i> CPU Temperature</h3>
                            <div class="value" id="temp-value">45.0</div>
                            <span class="unit">°C (simulated)</span>
                        </div>
                        <div class="status-card simulation">
                            <h3><i class="fas fa-clock"></i> Processing Time</h3>
                            <div class="value" id="process-value">32.5</div>
                            <span class="unit">ms (simulated)</span>
                        </div>
                        <div class="status-card simulation">
                            <h3><i class="fas fa-hdd"></i> Frames Served</h3>
                            <div class="value" id="frame-count">0</div>
                            <span class="unit">total</span>
                        </div>
                    </div>
                    
                    <!-- Video Stream -->
                    <div class="video-container fade-in" style="animation-delay: 0.4s">
                        <div class="video-wrapper">
                            <img id="video-feed" src="/video_feed" alt="Live Test Stream">
                            <div class="video-overlay">
                                <div class="spinner"></div>
                                <p style="margin-top: 20px;">Test stream loading...</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Pattern Info -->
                    <div class="pattern-info fade-in" style="animation-delay: 0.5s">
                        <h3><i class="fas fa-sync-alt"></i> Test Pattern Rotation (changes every 2 seconds)</h3>
                        <ul class="pattern-list">
                            <li><strong>Pattern 1:</strong> Single person detection (left side)</li>
                            <li><strong>Pattern 2:</strong> Single car detection (right side)</li>
                            <li><strong>Pattern 3:</strong> Multiple objects (person, car, chair, bottle)</li>
                            <li><strong>Pattern 4:</strong> Traffic scene (multiple people & cars)</li>
                            <li><strong>Pattern 5:</strong> Empty scene (no detections)</li>
                        </ul>
                    </div>
                    
                    <!-- Detections Panel -->
                    <div class="detections-panel fade-in" style="animation-delay: 0.6s">
                        <div class="panel-header">
                            <h2><i class="fas fa-bullseye"></i> Simulated Detections</h2>
                            <div class="detection-count" id="detection-count">0 objects</div>
                        </div>
                        <div class="detections-grid" id="detections-grid">
                            <!-- Detections will appear here -->
                            <div class="empty-state" style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-secondary);">
                                <i class="fas fa-search" style="font-size: 3rem; margin-bottom: 20px; opacity: 0.5;"></i>
                                <h3>No objects detected</h3>
                                <p>Waiting for simulation data...</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Footer -->
                    <footer class="connection-info fade-in" style="animation-delay: 0.8s">
                        <p><i class="fas fa-laptop-code"></i> Raspberry Pi 4 Detection System - Local Simulation</p>
                        <p><i class="fas fa-history"></i> Server Uptime: <span id="uptime">00:00:00</span></p>
                        <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.7;">
                            Auto-refreshes every 5 seconds | Testing optimized multi-client streaming
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
                                document.getElementById('temp-value').textContent = data.temperature.toFixed(1);
                                document.getElementById('process-value').textContent = data.processing_time.toFixed(1);
                                document.getElementById('frame-count').textContent = data.frame_count;
                                document.getElementById('client-count').textContent = data.active_clients;
                                document.getElementById('uptime').textContent = formatTime(data.server_uptime);
                                
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
                                        else if (det.label.toLowerCase().includes('car') || det.label.toLowerCase().includes('vehicle')) iconClass = 'vehicle';
                                        else if (det.label.toLowerCase().includes('bottle') || det.label.toLowerCase().includes('chair')) iconClass = 'object';
                                        
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
                                                    <div><i class="fas fa-sim-card"></i> SIMULATED</div>
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
                                            <p>Clear test pattern active</p>
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
                    
                    // Update every 1 second for faster feedback
                    setInterval(updateStats, 1000);
                    
                    // Auto-refresh page every 3 minutes
                    setTimeout(() => {
                        window.location.reload();
                    }, 180000);
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
        
        @self.app.route('/stats')
        def get_stats():
            with self.frame_lock:
                return jsonify({
                    'fps': self.detection_info['fps'],
                    'frame_count': self.detection_info['frame_count'],
                    'detections': self.detection_info['detections'],
                    'temperature': self.detection_info['temperature'],
                    'processing_time': self.detection_info['processing_time'],
                    'active_clients': self.detection_info['active_clients'],
                    'server_uptime': time.time() - self.detection_info['server_uptime'],
                    'total_frames_served': self.detection_info['total_frames_served']
                })
        
        @self.app.route('/server_info')
        def server_info():
            return jsonify({
                'ip': self._get_ip_address(),
                'port': self.port,
                'max_clients': self.max_clients,
                'status': 'simulation',
                'mode': 'local_test'
            })
    
    def _frame_distributor(self):
        """Optimized frame distributor for multiple clients"""
        last_frame_time = 0
        quality = 85
        
        while not self.stop_frame_thread.is_set():
            try:
                current_time = time.time()
                
                # Adjust frame rate based on client count
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
                                self.frame_buffer.get_nowait()
                            self.frame_buffer.put_nowait(jpeg_data)
                        except:
                            pass
                
                last_frame_time = current_time
                
            except Exception as e:
                print(f"Frame distributor error: {e}")
                time.sleep(0.1)
    
    def update_frame(self, frame, detections, fps=25.0, frame_count=0, temperature=45.0, processing_time=32.5):
        """Update frame with simulated detections - with Pi-like overlay"""
        frame_with_overlay = frame.copy()
        
        # Draw detection boxes
        detection_list = []
        for det in detections:
            label = det['label']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            # Color coding matching Pi version
            colors = {
                'person': (0, 100, 255),    # Orange-red
                'car': (255, 100, 0),       # Blue
                'chair': (100, 255, 0),     # Green
                'bottle': (255, 255, 100),  # Light yellow
            }
            color = colors.get(label.lower(), (255, 255, 100))
            
            # Draw bounding box
            cv2.rectangle(frame_with_overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
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
            
            detection_list.append(det)
        
        # Add info overlay (like Pi version)
        overlay_height = 90
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (20, 20, 20)
        
        # FPS and frame count
        fps_text = f"SIM FPS: {fps:5.1f} | Frame: {frame_count}"
        cv2.putText(overlay, fps_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Processing time
        proc_text = f"Process: {processing_time:5.1f}ms"
        cv2.putText(overlay, proc_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Temperature
        temp_color = (0, 255, 0)
        if temperature > 70:
            temp_color = (0, 255, 255)
        if temperature > 80:
            temp_color = (0, 0, 255)
        
        temp_text = f"Temp: {temperature:5.1f}°C (SIM)"
        cv2.putText(overlay, temp_text, (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, temp_color, 2)
        
        # "SIMULATION" watermark
        cv2.putText(overlay, "🧪 LOCAL SIMULATION - NO PI REQUIRED", 
                   (frame.shape[1] // 2 - 150, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 153, 0), 2)
        
        # Blend overlay
        alpha = 0.7
        frame_with_overlay[:overlay_height] = cv2.addWeighted(
            overlay, alpha, frame_with_overlay[:overlay_height], 1 - alpha, 0)
        
        # Update
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
        """Create placeholder frame"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Gradient background
        for i in range(img.shape[0]):
            color = int(30 + (i / img.shape[0]) * 40)
            img[i, :] = (color, color, color)
        
        cv2.putText(img, "🧪 LOCAL SIMULATION", (180, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 0), 2)
        cv2.putText(img, "Test stream starting...", (200, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return img
    
    def _create_overload_frame(self):
        """Create frame when server is at max capacity"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Orange gradient background
        for i in range(img.shape[0]):
            red = int(50 + (i / img.shape[0]) * 100)
            img[i, :] = (0, 100, red)
        
        cv2.putText(img, "TEST SERVER AT CAPACITY", (120, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        cv2.putText(img, f"Max test clients: {self.max_clients}", (180, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)
        
        cv2.putText(img, "Open another browser to test", (160, 300),
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
        print("🚀 OPTIMIZED LOCAL STREAMING SERVER STARTED")
        print("="*70)
        print(f"📡 On THIS computer, open:")
        print(f"   • http://localhost:{self.port}")
        print(f"   • http://{ip}:{self.port}")
        print(f"📱 On other devices, open:")
        print(f"   • http://{ip}:{self.port}")
        print("="*70)
        print("🎬 Test Patterns (rotating every 2 seconds):")
        print("   1. Single person detection")
        print("   2. Single car detection")
        print("   3. Multiple objects")
        print("   4. Traffic scene")
        print("   5. Empty scene")
        print("="*70)
        print("🔧 Features tested:")
        print("   • Optimized multi-client streaming")
        print("   • Dark UI with animations")
        print("   • Real-time detection display")
        print("   • Adaptive quality/FPS")
        print("="*70)
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        return server_thread
    
    def stop(self):
        """Stop the streaming server"""
        self.stop_frame_thread.set()
    
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

# ==================== MAIN TEST ====================
def run_local_test(duration_seconds=60):
    """Run complete local test"""
    print("🚀 Starting Optimized Pi 4 Streaming Simulation...")
    
    # Create mock system
    mock_system = MockDetectionSystem()
    
    # Start optimized streaming server
    streamer = LocalStreamingServer(port=5000, max_clients=20)
    streamer.run()
    
    # Give server time to start
    time.sleep(2)
    
    print("🎬 Starting optimized test sequence...")
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            # Get next test frame
            frame = mock_system.get_next_frame()
            
            # Simulate detection
            detections = mock_system.simulate_detection(frame)
            
            # Update streamer (throttled to ~15 FPS for demo)
            current_time = time.time()
            if current_time - getattr(mock_system, 'last_update', 0) >= 0.067:
                streamer.update_frame(
                    frame=frame,
                    detections=detections,
                    fps=mock_system.fps,
                    frame_count=mock_system.frame_count,
                    temperature=mock_system.temperature,
                    processing_time=32.5
                )
                mock_system.last_update = current_time
            
            # Print progress
            if frame_count % 20 == 0:
                pattern_names = ["PERSON", "CAR", "MULTIPLE", "TRAFFIC", "CLEAR"]
                current_pattern = pattern_names[(mock_system.pattern_index - 1) % 5]
                client_count = streamer.active_clients
                print(f"[{frame_count:4d}] Pattern: {current_pattern:12} | "
                      f"Detections: {len(detections):2d} | Clients: {client_count:2d}")
            
            frame_count += 1
            time.sleep(0.033)  # Faster for smoother test
            
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    
    streamer.stop()
    print("\n✅ Optimized local streaming test complete!")
    print(f"   Frames processed: {frame_count}")
    print(f"   Test duration: {time.time() - start_time:.1f}s")
    print(f"   Max clients tested: {streamer.max_clients}")
    print("\n📋 Verification Checklist:")
    print("   1. ✅ Open http://localhost:5000 in browser")
    print("   2. ✅ Verify dark theme with animations")
    print("   3. ✅ Verify test pattern rotation")
    print("   4. ✅ Verify detection boxes & confidence bars")
    print("   5. ✅ Open on phone: http://YOUR_IP:5000")
    print("   6. ✅ Test multiple browser windows")
    print("   7. ✅ Check adaptive quality with many clients")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=60, 
                       help='Test duration in seconds (default: 60)')
    parser.add_argument('--port', type=int, default=5000, 
                       help='Port for streaming server (default: 5000)')
    parser.add_argument('--clients', type=int, default=20,
                       help='Max concurrent clients (default: 20)')
    args = parser.parse_args()
    
    run_local_test(args.duration)