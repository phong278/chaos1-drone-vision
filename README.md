# Multi-Object Detection System for Raspberry Pi

A high-performance, thermal-aware object detection system optimized for Raspberry Pi 4, designed for drone deployment and real-time applications.

## Features

### Performance Optimizations

- **Multithreaded Processing**: Asynchronous inference pipeline using Python threading
- **Adaptive Frame Skipping**: Intelligent frame dropping to maintain target FPS
- **Configurable Input Resizing**: Reduce inference time with smaller input dimensions
- **FP16 Inference**: Half-precision support for faster computation
- **Minimal Memory Footprint**: Optimized buffers and data structures

### Thermal Management

- **Real-time Temperature Monitoring**: Continuously tracks CPU temperature
- **Auto-throttling**: Automatically reduces processing load when overheating
- **Hysteresis Control**: Prevents rapid on/off cycling
- **Perfect for Enclosed Spaces**: Essential for drone deployments

### Smart Storage Management

- **Automatic Cleanup**: Deletes oldest images when storage limit is reached
- **Configurable Size Limits**: Set maximum storage in MB
- **Thread-safe Operations**: Prevents corruption during concurrent access
- **JPEG Compression**: Adjustable quality settings to save space

### Detection Capabilities

- **80 COCO Classes**: Detects all standard COCO dataset objects
- **Priority System**: Focus on high-priority objects (people, phones, laptops, pets)
- **Object Tracking**: Smooth tracking with exponential smoothing
- **Confidence Filtering**: Adjustable confidence thresholds

### Monitoring & Logging

- **Real-time FPS Display**: Monitor system performance
- **Temperature Display**: Visual temperature feedback
- **File Logging**: Detailed logs with timestamps
- **Processing Time Stats**: Track inference performance

## Hardware Requirements

### Minimum (Tested Configuration)

- **Raspberry Pi 4 Model B** (4GB RAM recommended)
- **Camera**: USB webcam or Raspberry Pi Camera Module
- **Storage**: 8GB+ microSD card (Class 10 or better)
- **Power**: 5V 3A USB-C power supply

### Recommended for Drone Use

- **Active Cooling**: Heatsink + 5V fan
- **Storage**: 32GB+ microSD card for longer logging
- **Battery**: Sufficient capacity for extended operation

## Software Requirements

### Operating System

- Raspberry Pi OS (64-bit recommended)
- Ubuntu 20.04+ for ARM
- Any Linux distribution with V4L2 support

### Python Version

- Python 3.8 or higher

## Installation

### 1. System Setup

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

### 2. Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv detection_env
source detection_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install opencv-python==4.8.1.78
pip install ultralytics==8.0.196
pip install numpy==1.24.3

# For headless operation (optional)
pip install opencv-python-headless
```

### 3. Download the Script

```bash
# Clone or download the detection system
mkdir ~/detection_system
cd ~/detection_system

# Place the detection_system.py file here
# Create data directory
mkdir detection_data
```

### 4. Download YOLO Model

The script will automatically download the YOLOv8 model on first run, but you can pre-download:

```bash
# Download YOLOv8 Nano model (recommended for Pi)
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Configuration

Edit the `CONFIG` dictionary at the top of `detection_system.py`:

### Performance Modes

#### Balanced (Default)

```python
"performance": {
    "mode": "balanced",
    "throttle_fps": 15,
    "frame_skip": 1,
}
```

#### High Performance

```python
"performance": {
    "mode": "high_performance",
    "throttle_fps": 20,
    "frame_skip": 2,
}
```

#### Power Saving (Recommended for Drone)

```python
"performance": {
    "mode": "power_saving",
    "throttle_fps": 10,
    "frame_skip": 2,
}
```

### Storage Management

```python
"system": {
    "max_storage_mb": 500,  # Maximum storage for images
    "data_folder": "detection_data",
}
```

### Thermal Settings

```python
"system": {
    "thermal_throttle": True,  # Enable thermal management
    "temp_threshold": 75,      # Temperature limit in Celsius
}
```

### Camera Settings

```python
"camera": {
    "device_id": 0,           # Camera device (0 for default)
    "width": 640,
    "height": 480,
    "fps": 30,
    "flip_horizontal": False,
    "flip_vertical": False,
}
```

### Detection Settings

```python
"detection": {
    "model": "yolov8n.pt",    # Use nano model for Pi
    "confidence": 0.5,         # Detection confidence threshold
    "track_objects": True,     # Enable object tracking
}
```

## Usage

### Basic Usage

```bash
# Activate virtual environment
source detection_env/bin/activate

# Run the detection system
python3 detection_system.py

# Press 'q' in the OpenCV window to quit
```

### Headless Mode (No Display)

For drone deployment without a monitor:

```python
# Set in CONFIG:
"performance": {
    "mode": "high_performance",  # Disables visualization
}
```

### Running on Startup (Systemd)

Create a service file for automatic startup:

```bash
sudo nano /etc/systemd/system/detection.service
```

Add the following:

```ini
[Unit]
Description=Object Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/detection_system
ExecStart=/home/pi/detection_env/bin/python3 /home/pi/detection_system/detection_system.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable detection.service
sudo systemctl start detection.service
sudo systemctl status detection.service
```

### Running in Background

```bash
# Run in background with nohup
nohup python3 detection_system.py > detection.log 2>&1 &

# Check if running
ps aux | grep detection_system

# Stop the process
pkill -f detection_system.py
```

## Detected Object Classes

The system detects all 80 COCO classes:

**People & Animals**: person, cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Electronics**: tv, laptop, mouse, remote, keyboard, cell phone

**Furniture**: chair, couch, bed, dining table, toilet

**Kitchen Items**: bottle, wine glass, cup, fork, knife, spoon, bowl

**Sports**: sports ball, baseball bat, baseball glove, skateboard, surfboard, tennis racket, skis, snowboard, frisbee, kite

**Accessories**: backpack, umbrella, handbag, tie, suitcase

**Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Other**: potted plant, clock, vase, scissors, teddy bear, hair drier, toothbrush, book, and more

## Performance Benchmarks

### Raspberry Pi 4 (4GB)

| Mode | FPS | CPU Temp | Power |
|------|-----|----------|-------|
| Power Saving | 8-10 | ~65°C | Low |
| Balanced | 12-15 | ~70°C | Medium |
| High Performance | 15-18 | ~75°C | High |

**With YOLOv8n, 416x416 input, active cooling**

## Troubleshooting

### Camera Not Found

```bash
# List available cameras
ls -l /dev/video*

# Test camera
raspistill -o test.jpg  # For Pi Camera
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg  # For USB camera

# Update camera device_id in CONFIG if needed
```

### Low FPS / Performance Issues

1. **Enable Hardware Acceleration**:

   ```bash
   # Add to /boot/config.txt
   gpu_mem=128
   ```

2. **Reduce Input Size**:

   ```python
   "input_size": (320, 320),  # Even smaller
   ```

3. **Increase Frame Skip**:

   ```python
   "frame_skip": 3,  # Skip more frames
   ```

4. **Use Nano Model**:

   ```python
   "model": "yolov8n.pt",  # Smallest, fastest
   ```

### Overheating

1. **Install Active Cooling**: Add a heatsink and fan
2. **Lower Temperature Threshold**:

   ```python
   "temp_threshold": 70,
   ```

3. **Reduce FPS**:

   ```python
   "throttle_fps": 8,
   ```

4. **Improve Ventilation**: Ensure good airflow

### Storage Full

```bash
# Check storage usage
df -h

# Manually clear detection data
rm -rf detection_data/detection_*.jpg

# Reduce max storage in CONFIG
"max_storage_mb": 200,
```

### ImportError: No module named 'cv2'

```bash
# Reinstall OpenCV
pip uninstall opencv-python opencv-python-headless
pip install opencv-python==4.8.1.78

# Or for headless systems
pip install opencv-python-headless==4.8.1.78
```

## Optimization Tips for Drone Deployment

### 1. Power Efficiency

```python
CONFIG = {
    "performance": {
        "mode": "power_saving",
        "throttle_fps": 8,
        "frame_skip": 3,
    },
    "output": {
        "save_detections": False,  # Disable to save power
    }
}
```

### 2. Thermal Management

- Use the smallest possible enclosure with ventilation holes
- Mount Pi with heatsink exposed to airflow
- Consider the ICE Tower cooler for Pi 4

### 3. Storage

- Use high-endurance microSD cards
- Enable only necessary logging
- Set `save_on_detection: True` to save only relevant frames

### 4. Reliability

- Enable systemd service for auto-restart
- Log to file for debugging
- Monitor temperature continuously

## Advanced Features

### Custom Priority Objects

Edit the priority order to focus on specific objects:

```python
"tracking": {
    "priority_order": ["person", "car", "dog"],  # Your priorities
}
```

### Servo Control for Pan-Tilt

The system includes `control_servo()` method that calculates pan/tilt angles. Integrate with servo hardware:

```python
def control_servo(self, dx, dy):
    # Returns angles for servo control
    pan_angle, tilt_angle = self.control_servo(dx, dy)
    # Add your servo control code here
```

### Distance Estimation

Enable and calibrate distance estimation:

```python
"visualization": {
    "show_distance": True,
}

# Calibrate focal length in estimate_distance()
focal_length = 500  # Adjust based on your camera
```

## File Structure

```bash
detection_system/
├── detection_system.py      # Main script
├── detection_data/           # Output directory
│   ├── detection_*.jpg       # Saved detection images
│   └── detection_log_*.txt   # Log files
├── yolov8n.pt               # YOLO model (auto-downloaded)
└── README.md                # This file
```

## Contributing

Suggestions for improvement:

- Add support for multiple cameras
- Implement MQTT for remote monitoring
- Add GPS tagging for drone applications
- Support for other YOLO models (YOLOv9, YOLOv10)

## License & Intellectual Property

**Copyright © 2025. All Rights Reserved.**

### Proprietary Software Notice

This software and associated documentation files (the "Software") are proprietary and confidential. Unauthorized copying, distribution, modification, public display, or public performance of this Software, via any medium, is strictly prohibited.

**RESTRICTIONS:**

- ❌ No public distribution or sharing
- ❌ No commercial use without explicit written permission
- ❌ No modification or derivative works
- ❌ No reverse engineering or decompilation
- ✅ Private use for personal/internal purposes only

### Usage Rights

This Software is licensed, not sold. You are granted a limited, non-exclusive, non-transferable, revocable license to use this Software solely for your personal, non-commercial purposes.

For commercial licensing, partnerships, or other inquiries, please contact: [mabd8755@uni.sydney.edu.au]

## Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **OpenCV**: Computer vision library
- **Raspberry Pi Foundation**: Hardware platform

## Support

For issues or questions:

1. Check the Troubleshooting section
2. Review the configuration options
3. Check system logs in `detection_data/`
4. Monitor CPU temperature during operation

## Version History

### v2.0 (Current)

- Added multithreaded processing
- Implemented thermal management
- Smart storage management with auto-cleanup
- All 80 COCO classes support
- Raspberry Pi optimizations

### v1.0

- Initial release
- Basic detection functionality

---

**Note**: This system is optimized for Raspberry Pi 4 but will work on other platforms. Adjust settings based on your hardware capabilities.
