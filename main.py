from ultralytics import YOLO
from ultralytics.utils import LOGGER
import cv2

LOGGER.setLevel("ERROR")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# separate last-known positions
last_phone = None
last_person = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    frame_h, frame_w, _ = frame.shape
    cx_frame = frame_w // 2
    cy_frame = frame_h // 2

    cv2.circle(frame, (cx_frame, cy_frame), 5, (0,255,0), -1)

    # track which classes were detected
    phone_detected = False
    person_detected = False

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        if label not in ["cell phone", "person"]:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        obj_cx = int((x1 + x2) / 2)
        obj_cy = int((y1 + y2) / 2)

        dx = obj_cx - cx_frame
        dy = obj_cy - cy_frame

        # ---- PHONE LOGIC ----
        if label == "cell phone":
            phone_detected = True

            if last_phone != (dx, dy, obj_cx, obj_cy):
                print(f"[PHONE] dx={dx}, dy={dy} | center=({obj_cx}, {obj_cy})")
                last_phone = (dx, dy, obj_cx, obj_cy)

        # ---- PERSON LOGIC ----
        if label == "person":
            person_detected = True

            if last_person != (dx, dy, obj_cx, obj_cy):
                print(f"[PERSON] dx={dx}, dy={dy} | center=({obj_cx}, {obj_cy})")
                last_person = (dx, dy, obj_cx, obj_cy)

        # draw bounding box
        color = (0,255,0) if label == "cell phone" else (255,0,0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cv2.circle(frame, (obj_cx, obj_cy), 5, (0,0,255), -1)
        cv2.line(frame, (cx_frame, cy_frame), (obj_cx, obj_cy), color, 2)

        cv2.putText(frame, f"{label} {conf:.2f}",
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # reset individual trackers only if that class disappeared
    if not phone_detected:
        last_phone = None
    if not person_detected:
        last_person = None

    cv2.imshow("YOLO Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
