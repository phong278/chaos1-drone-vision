# gimbal_controller.py
import smbus2
import time
import threading
from queue import Queue
import math

class GimbalController:
    """Controls 3-axis gimbal via I2C/Serial"""
    
    def __init__(self, bus=1, address=0x52):
        self.bus = bus
        self.address = address
        self.i2c = None
        self.connected = False
        self.running = False
        self.target_queue = Queue(maxsize=10)
        self.current_angles = {'pitch': 0, 'roll': 0, 'yaw': 0}
        self.thread = None
        
    def connect(self):
        """Connect to gimbal controller via I2C"""
        try:
            self.i2c = smbus2.SMBus(self.bus)
            
            # Test connection
            self.i2c.write_byte(self.address, 0x00)
            time.sleep(0.1)
            
            print(f"Connected to gimbal controller at I2C address 0x{self.address:02x}")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Gimbal I2C connection failed: {e}")
            self.connected = False
            return False
    
    def _control_thread(self):
        """Thread that sends angle commands to gimbal"""
        while self.running:
            try:
                # Get target angles
                target = self.target_queue.get(timeout=0.1)
                
                # Limit angles for safety
                target['pitch'] = max(-90, min(30, target['pitch']))  # -90 to 30 degrees
                target['roll'] = max(-45, min(45, target['roll']))    # -45 to 45 degrees
                target['yaw'] = target['yaw'] % 360                   # 0 to 360 degrees
                
                # Send to gimbal controller
                self._send_angles(target)
                
                # Update current angles
                self.current_angles = target.copy()
                
            except Exception as e:
                # No target in queue, continue
                pass
    
    def _send_angles(self, angles):
        """Send angles to gimbal controller (I2C protocol)"""
        if not self.connected:
            return
        
        try:
            # Convert angles to controller format (0.1 degree resolution)
            pitch_int = int(angles['pitch'] * 10)
            roll_int = int(angles['roll'] * 10)
            yaw_int = int(angles['yaw'] * 10)
            
            # Create data packet
            data = [
                (pitch_int >> 8) & 0xFF, pitch_int & 0xFF,
                (roll_int >> 8) & 0xFF, roll_int & 0xFF,
                (yaw_int >> 8) & 0xFF, yaw_int & 0xFF
            ]
            
            # Send via I2C
            self.i2c.write_i2c_block_data(self.address, 0x00, data)
            
        except Exception as e:
            print(f"Error sending gimbal angles: {e}")
    
    def start(self):
        """Start gimbal control"""
        if not self.connected:
            if not self.connect():
                return False
        
        self.running = True
        self.thread = threading.Thread(target=self._control_thread, daemon=True)
        self.thread.start()
        print("Gimbal controller started")
        return True
    
    def stop(self):
        """Stop gimbal control"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("Gimbal controller stopped")
    
    def point_at_object(self, detection, frame_width, frame_height):
        """Calculate gimbal angles to center an object in frame"""
        if not detection:
            return None
        
        # Get object center
        x1, y1, x2, y2 = detection['bbox']
        obj_center_x = (x1 + x2) / 2
        obj_center_y = (y1 + y2) / 2
        
        # Calculate error from frame center
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        error_x = obj_center_x - frame_center_x
        error_y = obj_center_y - frame_center_y
        
        # Convert pixel error to angle (calibration needed)
        # These values depend on camera FOV and gimbal limits
        # 0.1 degree per pixel is a good starting point
        yaw_offset = -error_x * 0.1  # Negative because camera view is mirrored
        pitch_offset = error_y * 0.1
        
        # Calculate new target angles
        new_pitch = self.current_angles['pitch'] + pitch_offset
        new_yaw = (self.current_angles['yaw'] + yaw_offset) % 360
        
        # Create target angles (keep roll level)
        target = {
            'pitch': new_pitch,
            'roll': 0,  # Keep level
            'yaw': new_yaw
        }
        
        return target
    
    def set_target_angles(self, pitch=0, roll=0, yaw=0):
        """Set gimbal to specific angles"""
        target = {
            'pitch': float(pitch),
            'roll': float(roll),
            'yaw': float(yaw)
        }
        
        try:
            self.target_queue.put_nowait(target)
            return True
        except:
            # Queue full
            return False