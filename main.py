# ============================================
# NOTE:
# Speed values in this project are approximate/simulated.
# Real-world speed calculation requires camera calibration.
# This project is for demonstration of detection + tracking.
# ============================================

from ultralytics import YOLO
import cv2
import random
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("yolov8s.pt")
tracker = DeepSort(max_age=20)

cap = cv2.VideoCapture("video/test.mp4")

width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video (original quality)
out = cv2.VideoWriter(
    "output/output.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (width, height)
)

vehicle_classes = [2, 3, 5, 7]

track_speed = {}

bike_speeds = [55, 60, 66, 70]

speed_ranges = {
    "car": (80, 110),
    "bus": (60, 85),
    "van": (40, 40),
    "autorickshaw": (40, 60)
}

CAR_LIMIT = 100

frame_count = 0

cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    small = cv2.resize(frame, (640, 384))
    results = model(small)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in vehicle_classes and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = int(x1 * width / 640)
                y1 = int(y1 * height / 384)
                x2 = int(x2 * width / 640)
                y2 = int(y2 * height / 384)

                detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        l, t, w, h = map(int, track.to_ltrb())
        cls = track.det_class

        raw_label = model.names[cls] if cls is not None else "vehicle"

        # Force corrections
        if raw_label == "truck":
            label = "van"
        elif raw_label == "car":
            if h > 120 and w < 180:
                label = "van"
            else:
                label = "car"
        else:
            label = raw_label

        track_id = track.track_id

        # Speed logic
        if track_id not in track_speed:
            if label == "motorcycle":
                track_speed[track_id] = random.choice(bike_speeds)
            elif label in speed_ranges:
                track_speed[track_id] = random.randint(*speed_ranges[label])
            else:
                track_speed[track_id] = random.randint(40, 80)

        if label != "motorcycle":
            change = random.randint(-1, 2)
            speed = track_speed[track_id] + change

            if label in speed_ranges:
                min_s, max_s = speed_ranges[label]
                speed = max(min_s, min(speed, max_s))

            track_speed[track_id] = speed
        else:
            speed = track_speed[track_id]

        # Overspeed effect
        if label == "car" and speed > CAR_LIMIT:
            color = (0, 0, 255)
            cv2.rectangle(frame, (l, t), (l + w, t + h), color, 4)

            if frame_count % 10 < 5:
                cv2.putText(frame, "OVERSPEED!",
                            (l, t - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3)
        else:
            color = (0, 255, 0)
            cv2.rectangle(frame, (l, t), (l + w, t + h), color, 2)

        # Display text
        text = f"{label} approx {speed} km/h"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(frame, (l, t - th - 8), (l + tw, t), color, -1)

        cv2.putText(frame, text,
                    (l, t - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2)

    cv2.imshow("Output", frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
