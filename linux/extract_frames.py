import cv2
import os
from pathlib import Path

# ===== CONFIG =====
VIDEO_DIR = "../vids"          # directory containing videos
OUTPUT_DIR = "../frames/train"
FRAME_INTERVAL = 15            # save 1 frame every N frames
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
# ==================

os.makedirs(OUTPUT_DIR, exist_ok=True)

saved_id = 0

for video_path in sorted(Path(VIDEO_DIR).iterdir()):
    if video_path.suffix.lower() not in VIDEO_EXTS:
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}")
        continue

    frame_id = 0
    video_name = video_path.stem

    print(f"[INFO] Processing {video_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_INTERVAL == 0:
            out_name = f"{video_name}_{saved_id:06d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, frame)
            saved_id += 1

        frame_id += 1

    cap.release()

print(f"[DONE] Saved {saved_id} frames to {OUTPUT_DIR}")
