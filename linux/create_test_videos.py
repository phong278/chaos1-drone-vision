# create_test_videos.py
import cv2
import numpy as np

# Create test video with moving "objects"
def create_test_video(output_path="test_video.mp4", duration=10, fps=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    for i in range(duration * fps):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving "person" box
        x_pos = 100 + int(200 * np.sin(i / 10))
        cv2.rectangle(frame, (x_pos, 100), (x_pos + 100, 300), (0, 0, 255), -1)
        cv2.putText(frame, "PERSON", (x_pos, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Moving "car" box
        y_pos = 200 + int(100 * np.cos(i / 8))
        cv2.rectangle(frame, (400, y_pos), (550, y_pos + 100), (255, 0, 0), -1)
        cv2.putText(frame, "CAR", (400, y_pos - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Created test video: {output_path}")

create_test_video()