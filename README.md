# Multi-Object Detection System for Raspberry Pi

A high-performance, thermal-aware object detection system optimized for Raspberry Pi 4, featuring **real-time streaming, multi-client support, and professional web interface**. Designed for drone deployment and real-time monitoring applications.

## 🚀 Features

### **Real-Time Streaming & Web Interface** (NEW!)

- **Multi-Client Streaming**: Supports 10+ simultaneous viewers with adaptive quality
- **Professional Web UI**: Dark theme with animations, real-time stats, and detection display
- **Optimized Performance**: Adaptive frame rate (30→20 FPS) and quality (85→70%) based on client count
- **Connection Management**: Automatic overload protection and client limit enforcement
- **Cross-Platform Access**: View from any device - phones, tablets, laptops

### **Performance Optimizations**

- **Multithreaded Processing**: Asynchronous inference pipeline using Python threading
- **Adaptive Frame Skipping**: Intelligent frame dropping to maintain target FPS
- **Configurable Input Resizing**: Reduce inference time with smaller input dimensions
- **Minimal Memory Footprint**: Optimized buffers and data structures
- **Frame Buffering**: Efficient frame distribution for multiple clients

### **Thermal Management**

- **Real-time Temperature Monitoring**: Continuously tracks CPU temperature
- **Auto-throttling**: Automatically reduces processing load when overheating
- **Hysteresis Control**: Prevents rapid on/off cycling
- **Perfect for Enclosed Spaces**: Essential for drone deployments

### **Smart Storage Management**

- **Automatic Cleanup**: Deletes oldest images when storage limit is reached
- **Configurable Size Limits**: Set maximum storage in MB
- **Thread-safe Operations**: Prevents corruption during concurrent access
- **JPEG Compression**: Adjustable quality settings to save space

### **Detection Capabilities**

- **80 COCO Classes**: Detects all standard COCO dataset objects
- **Priority System**: Focus on high-priority objects (people, phones, laptops, pets)
- **Confidence Filtering**: Adjustable confidence thresholds
- **Real-time Overlay**: Detection boxes and labels on video stream

### **Monitoring & Logging**

- **Real-time FPS Display**: Monitor system performance
- **Temperature Display**: Visual temperature feedback
- **File Logging**: Detailed logs with timestamps
- **Processing Time Stats**: Track inference performance
- **Web Interface Stats**: Live updates of all system metrics

## 📊 Hardware Requirements

### **Minimum (Tested Configuration)**

- **Raspberry Pi 4 Model B** (4GB RAM recommended)
- **Camera**: USB webcam or Raspberry Pi Camera Module
- **Storage**: 8GB+ microSD card (Class 10 or better)
- **Power**: 5V 3A USB-C power supply
- **Network**: WiFi or Ethernet for streaming

### **Recommended for Drone Use**

- **Active Cooling**: Heatsink + 5V fan
- **Storage**: 32GB+ microSD card for longer logging
- **Battery**: Sufficient capacity for extended operation
- **Reliable WiFi**: For live streaming from drone

## 💻 Software Requirements

### **Operating System**

- Raspberry Pi OS (64-bit recommended)
- Ubuntu 20.04+ for ARM
- Any Linux distribution with V4L2 support

### **Python Version**

- Python 3.8 or higher

## 📥 Installation

### **1. System Setup**

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-opencv libopencv-dev
sudo apt install -y python3-numpy libatlas-base-dev

# Enable camera (if using Pi Camera Module)
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
```

### **2. Python Dependecies**

```bash
# Create virtual environment (recommended)
python3 -m venv detection_env
source detection_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install flask==3.0.0

# For production streaming (optional but recommended)
pip install gunicorn gevent
```

### **3. Download the System**

```bash
# Create project directory
mkdir ~/pi_detection_system
cd ~/pi_detection_system

# Download all required files:
# - main.py (detection system)
# - streaming_server.py (web interface)
# - config_loader.py (configuration management)
# - config.json (configuration)
# - models/ (YOLO model files)

# Create directories
mkdir -p detection_data/models
```

### **4. Model Setup**

The system will automatically download YOLOv4-tiny on first run, but you can pre-download:

```bash
cd models
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

## ⚙️ Configuration Management

The system includes a comprehensive configuration management utility:

### Configuration Commands

```bash
# Show current configuration
python manage_config.py show

# Create default config
python manage_config.py create

# Create test configuration
python manage_config.py create-test

# Edit a configuration value
python manage_config.py edit system.enable_streaming true
python manage_config.py edit performance.throttle_fps 20
python manage_config.py edit system.max_streaming_clients 25

# Validate configuration
python manage_config.py validate

# Compare with defaults
python manage_config.py diff
```

### Configuration Structure

```json
{
  "system": {
    "data_folder": "detections",
    "max_storage_mb": 1024,
    "temp_threshold": 75,
    "log_to_file": true,
    "enable_streaming": true,
    "streaming_port": 5000,
    "max_streaming_clients": 15
  },
  "camera": {
    "device_id": 0,
    "backend": "CAP_V4L2",
    "width": 640,
    "height": 480,
    "fps": 30,
    "buffer_size": 2
  },
  "detection": {
    "model_cfg": "models/yolov4-tiny.cfg",
    "model_weights": "models/yolov4-tiny.weights",
    "labels": "models/coco.names",
    "confidence": 0.4
  },
  "performance": {
    "use_threading": true,
    "frame_buffer_size": 2,
    "frame_skip": 0,
    "throttle_fps": 20,
    "input_size": [416, 416],
    "conf_threshold": 0.25,
    "nms_threshold": 0.4
  },
  "output": {
    "save_detections": true,
    "save_on_detection": false,
    "save_interval": 5,
    "image_quality": 85,
    "console_log": true,
    "file_log": true,
    "print_detections": true
  }
}
```

## Usage

### Basic Usage with Streaming

```bash
# Activate virtual environment
source detection_env/bin/activate

# Run with streaming enabled (default)
python main.py

# Access the web interface from:
# - On Pi: http://localhost:5000
# - On network: http://YOUR_PI_IP:5000
```

## Running on Startup

```bash
sudo nano /etc/systemd/system/detection.service
```

```ini
[Unit]
Description=Pi Detection System with Streaming
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pi_detection_system
Environment="PATH=/home/pi/detection_env/bin"
ExecStart=/home/pi/detection_env/bin/python /home/pi/pi_detection_system/main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable detection.service
sudo systemctl start detection.service
sudo systemctl status detection.service

# View logs
sudo journalctl -u detection.service -f
```

## Accessing the Stream

### On Local Network

```bash
# Find Pi IP address
hostname -I

# Access from any device on same network
# Desktop: http://192.168.1.100:5000
# Phone: http://192.168.1.100:5000
# Tablet: http://192.168.1.100:5000
```

### Remote Access

1. Port Forwarding: Forward port 5000 to your Pi's IP

2. Dynamic DNS: Use services like DuckDNS or No-IP

3. Access URL: `http://your-domain.com:5000` or `http://your-public-ip:5000`

## Detected Object Classes

The system detects all 80 COCO classes with real-time overlay on stream:

People & Animals: person, cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe

Vehicles: bicycle, car, motorcycle, airplane, bus, train, truck, boat

Electronics: tv, laptop, mouse, remote, keyboard, cell phone

Furniture: chair, couch, bed, dining table, toilet

Kitchen Items: bottle, wine glass, cup, fork, knife, spoon, bowl

Sports: sports ball, baseball bat, baseball glove, skateboard, surfboard, tennis racket, skis, snowboard, frisbee, kite

Food: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

## Testing & Development

```bash
# Test streaming server on your laptop
python test_streaming_server_local.py

# Test with specific duration
python test_streaming_server_local.py --duration 120 --clients 30

# Features tested:
# 1. Web interface with dark theme
# 2. Multi-client streaming
# 3. Detection display
# 4. Adaptive quality/FPS
```

## Troubleshooting

### **Streaming Issues**

```bash
# Check if server is running
ps aux | grep python | grep streaming

# Check port availability
sudo netstat -tlnp | grep :5000

# Test streaming locally
curl http://localhost:5000/video_feed

# Increase max clients if needed
python manage_config.py edit system.max_streaming_clients 25
```

## File Structure

```bash
pi_detection_system/
├── main.py                      # Main detection system
├── streaming_server.py          # Streaming server with web UI
├── config_loader.py             # Configuration management
├── manage_config.py             # Configuration utility
├── test_streaming_server_local.py # Local testing
├── config.json                  # Configuration file
├── detection_data/              # Output directory
│   ├── detection_*.jpg          # Saved detection images
│   └── detection_log_*.txt      # Log files
├── models/                      # Model files
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
│   └── coco.names
└── README.md                    # This file
```

## License & Intellectual Property

Copyright © 2025. All Rights Reserved.

### Proprietary Software Notice

This software and associated documentation files (the "Software") are proprietary and confidential. Unauthorized copying, distribution, modification, public display, or public performance of this Software, via any medium, is strictly prohibited.

**Restrictions:**

❌ No public distribution or sharing

❌ No commercial use without explicit written permission

❌ No modification or derivative works

❌ No reverse engineering or decompilation

✅ Private use for personal/internal purposes only

### Usage Rights

This Software is licensed, not sold. You are granted a limited, non-exclusive, non-transferable, revocable license to use this Software solely for your personal, non-commercial purposes.

For commercial licensing, partnerships, or other inquiries, please contact: [mabd8755@uni.sydney.edu.au]

## Version History

### V3.0 (Current)

- Added real-time streaming with web interface
- Multi-client support with adaptive quality
- Configuration management utility
- Local testing system without Pi hardware
- Professional dark theme UI

### V2.0

- Added multithreaded processing
- Implemented thermal management
- Smart storage management with auto-cleanup
- All 80 COCO classes support
- Raspberry Pi optimizations

### V1.0

- Intial release with basic object detection
