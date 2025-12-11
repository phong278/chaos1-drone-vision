#!/usr/bin/env python3
"""
Configuration management utility
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_loader import ConfigLoader
import json

def print_help():
    print("""
Configuration Management Utility
===============================

Commands:
  help              Show this help message
  show              Show current configuration
  create            Create default config.json
  create-test       Create test configuration
  edit <key> <value> Edit a configuration value
  validate          Validate current configuration
  diff              Compare with default configuration

Examples:
  python manage_config.py show
  python manage_config.py create
  python manage_config.py edit system.enable_streaming false
  python manage_config.py edit performance.throttle_fps 15
""")

def edit_config(key_path, value):
    """Edit a configuration value"""
    config = ConfigLoader.load()
    
    # Parse key path (e.g., "system.enable_streaming")
    keys = key_path.split('.')
    current = config
    
    # Navigate to the nested key
    for key in keys[:-1]:
        if key not in current:
            print(f"Invalid key path: {key}")
            return
        current = current[key]
    
    # Convert value to appropriate type
    last_key = keys[-1]
    if last_key not in current:
        print(f"Invalid key: {last_key}")
        return
    
    # Type conversion based on existing value
    old_value = current[last_key]
    if isinstance(old_value, bool):
        if value.lower() in ['true', '1', 'yes', 'y']:
            new_value = True
        elif value.lower() in ['false', '0', 'no', 'n']:
            new_value = False
        else:
            print(f"Invalid boolean value: {value}")
            return
    elif isinstance(old_value, int):
        try:
            new_value = int(value)
        except ValueError:
            print(f"Invalid integer: {value}")
            return
    elif isinstance(old_value, float):
        try:
            new_value = float(value)
        except ValueError:
            print(f"Invalid float: {value}")
            return
    else:
        new_value = value
    
    # Update and save
    current[last_key] = new_value
    ConfigLoader.save_current(config)
    print(f"Updated {key_path}: {old_value} → {new_value}")

def validate_config():
    """Validate the current configuration"""
    config = ConfigLoader.load()
    
    print("Validating configuration...")
    
    issues = []
    
    # Check required model files
    required_files = [
        config["detection"]["model_cfg"],
        config["detection"]["model_weights"],
        config["detection"]["labels"]
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"Missing file: {file}")
    
    # Check data folder
    data_folder = config["system"]["data_folder"]
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' doesn't exist (will be created)")
    
    # Check streaming port range
    port = config["system"]["streaming_port"]
    if port < 1024 or port > 65535:
        issues.append(f"Invalid streaming port: {port} (must be 1024-65535)")
    
    # Check FPS
    fps = config["camera"]["fps"]
    if fps <= 0 or fps > 60:
        issues.append(f"Invalid FPS: {fps} (should be 1-60)")
    
    # Check confidence threshold
    conf = config["detection"]["confidence"]
    if conf < 0 or conf > 1:
        issues.append(f"Invalid confidence threshold: {conf} (should be 0.0-1.0)")
    
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Configuration is valid")
        return True

def show_config():
    """Display current configuration"""
    config = ConfigLoader.load()
    
    print("Current Configuration:")
    print("="*60)
    
    def print_section(section, data, indent=0):
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_section(section, value, indent + 1)
            else:
                print(f"{prefix}{key}: {value}")
    
    for section, data in config.items():
        print(f"\n{section.upper()}:")
        print("-" * 40)
        print_section(section, data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "help":
        print_help()
    elif command == "show":
        show_config()
    elif command == "create":
        ConfigLoader.save_default()
    elif command == "create-test":
        ConfigLoader.create_test_config()
    elif command == "validate":
        validate_config()
    elif command == "diff":
        from config_loader import ConfigLoader
        current = ConfigLoader.load()
        default = ConfigLoader.DEFAULT_CONFIG
        
        print("Differences from default configuration:")
        # Simple diff implementation
        import json
        current_json = json.dumps(current, sort_keys=True, indent=2)
        default_json = json.dumps(default, sort_keys=True, indent=2)
        
        if current_json == default_json:
            print("Configuration matches defaults")
        else:
            print("Configuration differs from defaults")
    elif command == "edit" and len(sys.argv) == 4:
        edit_config(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)