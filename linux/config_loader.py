# config_loader.py
import json
import os
from pathlib import Path

class ConfigLoader:
    """Load and manage configuration from JSON file"""
    
    DEFAULT_CONFIG = {
        "system": {
            "platform": "raspberry",
            "log_to_file": True,
            "data_folder": "detection_data",
            "max_storage_mb": 500,
            "thermal_throttle": True,
            "temp_threshold": 75,
            "enable_streaming": True,
            "streaming_port": 5000,
            "max_streaming_clients": 5,
        },
        "performance": {
            "mode": "balanced",
            "throttle_fps": 10,
            "frame_skip": 1,
            "resize_input": True,
            "input_size": [320, 320],
            "use_threading": True,
            "frame_buffer_size": 2,
            "nms_threshold": 0.45,
            "conf_threshold": 0.4,
        },
        "camera": {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 15,
            "flip_horizontal": False,
            "flip_vertical": False,
            "backend": "CAP_V4L2",
            "buffer_size": 1,
        },
        "detection": {
            "model_cfg": "yolov4-tiny.cfg",
            "model_weights": "yolov4-tiny.weights",
            "labels": "coco.names",
            "confidence": 0.4,
            "max_classes": 80,
        },
        "output": {
            "console_log": True,
            "file_log": True,
            "save_detections": True,
            "save_interval": 5,
            "save_on_detection": True,
            "image_quality": 70,
            "print_detections": True,
        }
    }
    
    @staticmethod
    def load(config_path="config.json"):
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    print(f"✅ Loaded configuration from {config_path}")
                    return ConfigLoader._merge_configs(ConfigLoader.DEFAULT_CONFIG, loaded_config)
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing {config_path}: {e}")
                print("⚠️ Using default configuration")
                return ConfigLoader.DEFAULT_CONFIG
        else:
            print(f"⚠️ Config file {config_path} not found, creating default...")
            ConfigLoader.save_default(config_path)
            return ConfigLoader.DEFAULT_CONFIG
    
    @staticmethod
    def _merge_configs(default, custom):
        """Deep merge two configuration dictionaries"""
        merged = default.copy()
        
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @staticmethod
    def save_default(config_path="config.json"):
        """Save default configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(ConfigLoader.DEFAULT_CONFIG, f, indent=2)
        print(f"📝 Created default configuration at {config_path}")
    
    @staticmethod
    def save_current(config, config_path="config.json"):
        """Save current configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Saved current configuration to {config_path}")
    
    @staticmethod
    def create_test_config(config_path="config_test.json"):
        """Create a test configuration for development"""
        test_config = ConfigLoader.DEFAULT_CONFIG.copy()
        test_config["system"]["enable_streaming"] = True
        test_config["performance"]["throttle_fps"] = 5
        test_config["camera"]["backend"] = "CAP_ANY"
        test_config["output"]["save_interval"] = 2
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        print(f"🧪 Created test configuration at {config_path}")
        return test_config

# For backward compatibility
def load_config(config_path="config.json"):
    return ConfigLoader.load(config_path)