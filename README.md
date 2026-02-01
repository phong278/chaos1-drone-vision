# Multi-Object Detection, Tracking & Servo Control System for Raspberry Pi 5

A real-time **vision-based detection, tracking, servo control, and MJPEG streaming system** designed for Raspberry Pi 5. The system uses a YOLOv8 ONNX model with CPU inference and precision motion control logic to physically track detected targets using pan/tilt servos.

Designed for **surveillance, robotics, autonomous platforms, and research prototypes** where lightweight, low-latency, on-device inference is required.

---

## Overview

This system performs real-time object detection on a live camera feed, tracks a selected target with a **state-based precision controller**, and streams annotated video over the network using a **native MJPEG HTTP server**.

Key capabilities:

* Real-time YOLOv8 object detection (ONNX Runtime)
* Precision pan/tilt servo tracking (vision-only, no IMU)
* Motion-aware target locking (no jitter when target is stationary)
* PD-controlled servo movement with adaptive gains
* Lightweight MJPEG streaming over HTTP
* Designed and tuned for Raspberry Pi 5

---

## System Architecture

**Pipeline:**

Camera → YOLOv8 ONNX Inference → Target Selection →
Tracking State Machine → PD Servo Controller →
MJPEG Stream Output

The system is intentionally single-process and CPU-only to ensure deterministic timing and stability on embedded hardware.

---

## Tracking Logic

The tracker operates as a **state machine** to improve mechanical precision and stability:

### Tracking States

* **SEARCHING** – No valid target detected
* **CENTERING** – Target detected, servos actively center
* **LOCKED** – Target centered and stationary (servos frozen)

### Motion Awareness

* Target motion is estimated using recent image-space position history
* Servos only move when:

  * The target drifts off-center, or
  * The target begins moving again

This prevents micro-oscillation caused by detector noise.

---

## Servo Control

* Hardware controlled using `gpiozero` + `lgpio`
* Vision-only PD controller (Proportional + Derivative)
* Adaptive gains depending on tracking state

---

## Streaming

* Native MJPEG server
* Streams annotated frames over HTTP
* Accessible from any browser on the local network

Example:

`http://192.168.1.42:5000`

Designed for:

* Low latency
* Minimal CPU overhead
* Easy integration with dashboards or monitoring systems

---

## Dataset & Classes

### Dataset YAML (data.yml)

```yaml
path: ..

train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images

nc: 9

names:
  0: Artillery
  1: Missile
  2: Radar
  3: M. Rocket Launcher
  4: Soldier
  5: Tank
  6: Vehicle
  7: Aircraft
  8: Ships
```

### Dataset Composition

The dataset is built on **KIIT-MILA** as the primary source of ground-based military targets, with additional datasets incorporated to expand aerial and maritime coverage.

**Base Dataset**

* KIIT-MILA (Military Imagery for Target Detection)
  [https://www.kaggle.com/datasets/sudipchakrabarty/kiit-mita](https://www.kaggle.com/datasets/sudipchakrabarty/kiit-mita)

Used for tanks, vehicles, artillery, radar systems, and personnel.

**Extended Datasets**

* Military Aircraft Detection Dataset
  [https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)

* Ships in Aerial Images Dataset
  [https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images](https://www.kaggle.com/datasets/siddharthkumarsah/ships-in-aerial-images)

All datasets were converted to YOLO format, unified under a single class mapping, and merged into a balanced train/validation/test split.

---

## Runtime Characteristics

* CPU-only inference (no GPU / NPU required)
* Tuned for real-time performance at low resolutions
* Stable servo behavior even under noisy detections
* Designed for continuous unattended operation

---

## Requirements 🔧

* Python 3.10+ (3.11 recommended)
* See `linux/requirementsPI.txt` for exact dependencies (e.g. `onnxruntime`, `opencv-python`, `numpy`, `gpiozero`, `lgpio`)

## Quick Start ✅

1. Install dependencies: `pip install -r linux/requirementsPI.txt`
2. Run the detector and tracker: `python3 linux/detection.py`
3. View the stream in a browser: `http://<pi-ip>:5000`

Optional examples:
* `python3 linux/detect_video.py --source /dev/video0`
* `python3 linux/detect_images.py --source dataset/test/images`

---

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

For commercial licensing, partnerships, or other inquiries, please contact:

📧 [mabd8755@uni.sydney.edu.au](mailto:mabd8755@uni.sydney.edu.au)

---

## Version History

### V4.5 (Current)

* Documentation and quick-start additions for easier setup
* Minor tracking & stability tuning (conservative predictive trust, damping, deadzone)
* Safer servo defaults and clearer debug/status output

### V4.0 (Previous)

* YOLOv8 ONNX Runtime (CPU inference)
* Precision vision-only servo tracking (pan/tilt)
* Motion-aware target locking
* PD-controlled servo movement with adaptive gains
* Native MJPEG HTTP streaming server
* Raspberry Pi 5 optimized

### V3.0

* Added real-time streaming with web interface
* Multi-client support with adaptive quality
* Configuration management utility
* Local testing system without Pi hardware
* Professional dark theme UI

### V2.0

* Added multithreaded processing
* Implemented thermal management
* Smart storage management with auto-cleanup
* All 80 COCO classes support
* Raspberry Pi optimizations

### V1.0

* Initial release with basic object detection
