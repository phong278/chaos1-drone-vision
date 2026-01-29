#!/usr/bin/env python3
import time
from gpiozero import Servo, Device
from gpiozero.pins.lgpio import LGPIOFactory
import sys
import termios
import tty

# ================= GPIO SETUP =================
Device.pin_factory = LGPIOFactory()

pan_servo  = Servo(18, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
tilt_servo = Servo(13, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Center servos
pan_pos = 0.0
tilt_pos = 0.0
pan_servo.value = pan_pos
tilt_servo.value = tilt_pos

STEP = 0.1  # Amount to move per keypress

# ================= KEYBOARD INPUT =================
def get_key():
    """Read a single keypress from terminal."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("Control servos with WASD keys (W=up, S=down, A=left, D=right). Q to quit.")

# ================= MAIN LOOP =================
while True:
    key = get_key().lower()

    if key == 'w':
        tilt_pos -= STEP
    elif key == 's':
        tilt_pos += STEP
    elif key == 'a':
        pan_pos -= STEP
    elif key == 'd':
        pan_pos += STEP
    elif key == 'q':
        break
    else:
        continue  # ignore other keys

    # Clamp values
    pan_pos = max(-1.0, min(1.0, pan_pos))
    tilt_pos = max(-1.0, min(1.0, tilt_pos))

    # Apply servo values
    pan_servo.value = pan_pos
    tilt_servo.value = tilt_pos

    print(f"Pan: {pan_pos:+.2f}, Tilt: {tilt_pos:+.2f}")
    time.sleep(0.05)

print("Exiting manual control...")
