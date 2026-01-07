# Multi-Object Detection & Streaming System for Raspberry Pi 5

A real-time object detection and MJPEG streaming system optimized for Raspberry Pi 5, using a custom-trained YOLOv8 model deployed via OpenVINO for efficient edge inference.

Designed for defense, surveillance, robotics, and drone-based monitoring applications.

## Overview

This system performs real-time detection on live camera input, overlays detections, and streams the processed video over the network using a lightweight MJPEG HTTP server.

Key goals:

- Run efficiently on Raspberry Pi 5

- Use custom military dataset

- Support OpenVINO inference

- Minimal latency, minimal dependencies

- Network-accessible live stream

## Dataset & Classes

Dataset Source

KIIT-MITA Dataset (Kaggle)

`https://www.kaggle.com/datasets/sudipchakrabarty/kiit-mita`

Dataset YAML

```yaml
path: dataset

train: train/images
val: valid/images
test: test/images

nc: 7
names:
  - Artillery
  - Missile
  - Radar
  - M. Rocket Launcher
  - Soldier
  - Tank
  - Vehicle
```

### Supported Classes

| ID | Class |
| ----------- | ----------- |
| 0 | Artillery |
| 1 | Missile |
| 2 | Radar |
| 3 | M. Rocket Launcher |
| 4 | Soldier |
| 5 | Tank |
| 6 | Vehicle |

## Software Requirements

- Raspberry Pi OS 64-bit

- Python 3.9+

- OpenVINO Runtime

- OpenCV with V4L2 support

## Configuration

Default Runtime Configuration (in code)

```python
config = {
    'system': {
        'enable_streaming': True,
        'streaming_port': 5000
    },
    'performance': {
        'throttle_fps': 15,
        'frame_skip': 2,
        'inference_size': 320
    },
    'camera': {
        'device_id': 0,
        'width': 1280,
        'height': 720,
        'fps': 15
    },
    'detection': {
        'confidence': 0.4
    }
}
```

## License & Intellectual Property

Copyright © 2025 Muhammad Abdullah and Phong Ngo.

All rights jointly owned. No license or transfer without written consent of both parties.

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

### V4.0 (Current)

- YOLOv8 + OpenVINO pipeline
- Custom KIIT-MITA dataset
- Raspberry Pi 5 optimized
- Lightweight MJPEG streaming
- ONNX → OpenVINO deployment
- Changed streaming from flask to native HTTP server

### V3.0

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
