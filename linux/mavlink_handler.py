# mavlink_handler.py
import threading
from queue import Queue
from pymavlink import mavutil
import time

class MavlinkHandler:
    """Handles MAVLink communication with flight controller"""
    
    def __init__(self, port='/dev/ttyAMA0', baud=921600):
        self.port = port
        self.baud = baud
        self.master = None
        self.connected = False
        self.running = False
        self.telemetry_queue = Queue(maxsize=100)
        self.command_queue = Queue(maxsize=100)
        self.thread = None
        
    def connect(self):
        """Connect to flight controller"""
        try:
            self.master = mavutil.mavlink_connection(
                self.port, 
                baud=self.baud,
                source_system=255,  # Pi system ID
                source_component=0,  # Pi component ID
                dialect='ardupilotmega'
            )
            
            print(f"Waiting for FC heartbeat on {self.port}...")
            self.master.wait_heartbeat(timeout=5)
            print(f"Connected to FC: System {self.master.target_system}")
            
            # Request data stream
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # 10 Hz
                1    # Start sending
            )
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"MAVLink connection failed: {e}")
            self.connected = False
            return False
    
    def _communication_thread(self):
        """Thread that handles MAVLink communication"""
        while self.running:
            try:
                # Send queued commands
                if not self.command_queue.empty():
                    cmd = self.command_queue.get_nowait()
                    self.master.mav.send(cmd)
                
                # Read telemetry
                msg = self.master.recv_match(blocking=False, timeout=0.1)
                if msg:
                    self.telemetry_queue.put(msg)
                    
                    # Log important messages
                    msg_type = msg.get_type()
                    if msg_type == 'HEARTBEAT':
                        mode = mavutil.mode_string_v10(msg)
                        print(f"FC Mode: {mode}, Armed: {msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED}")
                    elif msg_type == 'GLOBAL_POSITION_INT':
                        lat = msg.lat / 1e7
                        lon = msg.lon / 1e7
                        alt = msg.alt / 1000
                        print(f"GPS: Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt:.1f}m")
                        
            except Exception as e:
                print(f"MAVLink thread error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start MAVLink communication"""
        if not self.connected:
            if not self.connect():
                return False
        
        self.running = True
        self.thread = threading.Thread(target=self._communication_thread, daemon=True)
        self.thread.start()
        print("MAVLink communication started")
        return True
    
    def stop(self):
        """Stop MAVLink communication"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("MAVLink communication stopped")
    
    def send_gimbal_command(self, pitch=0, roll=0, yaw=0):
        """Send gimbal control command to FC"""
        if not self.connected:
            return False
        
        try:
            # Convert angles to centi-degrees
            pitch_cd = int(pitch * 100)
            roll_cd = int(roll * 100)
            yaw_cd = int(yaw * 100)
            
            # Send mount control command
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
                0,          # Confirmation
                pitch,      # Pitch in degrees
                roll,       # Roll in degrees
                yaw,        # Yaw in degrees
                0, 0, 0, 0, 0  # Unused parameters
            )
            
            # Also send direct mount control
            self.master.mav.mount_control_send(
                self.master.target_system,
                self.master.target_component,
                pitch_cd,
                roll_cd,
                yaw_cd,
                0  # Save position
            )
            
            return True
            
        except Exception as e:
            print(f"Error sending gimbal command: {e}")
            return False
    
    def get_latest_telemetry(self):
        """Get latest telemetry data"""
        telemetry = {}
        while not self.telemetry_queue.empty():
            msg = self.telemetry_queue.get_nowait()
            msg_type = msg.get_type()
            
            if msg_type == 'ATTITUDE':
                telemetry['attitude'] = {
                    'roll': msg.roll,
                    'pitch': msg.pitch,
                    'yaw': msg.yaw
                }
            elif msg_type == 'GLOBAL_POSITION_INT':
                telemetry['position'] = {
                    'lat': msg.lat / 1e7,
                    'lon': msg.lon / 1e7,
                    'alt': msg.alt / 1000,
                    'relative_alt': msg.relative_alt / 1000
                }
            elif msg_type == 'VFR_HUD':
                telemetry['hud'] = {
                    'airspeed': msg.airspeed,
                    'groundspeed': msg.groundspeed,
                    'heading': msg.heading,
                    'throttle': msg.throttle,
                    'alt': msg.alt
                }
        
        return telemetry