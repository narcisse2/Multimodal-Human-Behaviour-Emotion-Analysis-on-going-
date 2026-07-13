import cv2
from ultralytics import YOLO

# Load YOLO Pose model once
model = YOLO("yolov8n-pose.pt")


def detect_pose(frame):
    """
    Draw:
    - Person box
    - Shoulders
    - Elbows
    - Wrists
    """

    results = model(frame, verbose=False)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        if result.keypoints is not None:
            persons = result.keypoints.xy.cpu().numpy()

            for person in persons:
                if len(person) < 11:
                    continue

                joints = [
                    (person[5], (0, 0, 255)),
                    (person[6], (0, 0, 255)),
                    (person[7], (255, 0, 0)),
                    (person[8], (255, 0, 0)),
                    (person[9], (0, 255, 255)),
                    (person[10], (0, 255, 255)),
                ]

                for point, color in joints:
                    x, y = point

                    if x > 0 and y > 0:
                        cv2.circle(
                            frame,
                            (int(x), int(y)),
                            6,
                            color,
                            -1
                        )

    return frame


def extract_pose_features(frame):
    results = model(frame, verbose=False)

    boxers = []

    for result in results:
        if result.boxes is None or result.keypoints is None:
            continue

        boxes = result.boxes
        keypoints = result.keypoints.xy.cpu().numpy()

        for box, person_keypoints in zip(boxes, keypoints):
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0 or conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            body_points = []

            for point in person_keypoints:
                x, y = point

                if x > 0 and y > 0:
                    body_points.append((float(x), float(y)))

            if not body_points:
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            boxers.append(
                {
                    "box": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "keypoints": body_points,
                }
            )

    boxers = sorted(boxers, key=lambda boxer: boxer["center"][0])

    return boxers


def calculate_movement_intensity(current_boxers, previous_boxers):
    movements = []

    for index, current_boxer in enumerate(current_boxers):
        if index >= len(previous_boxers):
            movements.append(0.0)
            continue

        previous_boxer = previous_boxers[index]

        current_x, current_y = current_boxer["center"]
        previous_x, previous_y = previous_boxer["center"]

        dx = current_x - previous_x
        dy = current_y - previous_y

        movement = (dx ** 2 + dy ** 2) ** 0.5

        movements.append(round(movement, 2))

    return movements
