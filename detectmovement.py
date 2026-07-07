import cv2
from ultralytics import YOLO

# Load YOLO Pose model once
model = YOLO("yolov8n-pose.pt")


def detect_pose(frame):
    """
    Draw:
    - Person box (green)
    - Shoulders (red)
    - Elbows (blue)
    - Wrists (yellow)
    """

    results = model(frame, verbose=False)

    for result in results:

        # Person boxes
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
                        (0,255,0),
                        2
                    )

        # Pose keypoints
        if result.keypoints is not None:

            persons = result.keypoints.xy.cpu().numpy()

            for person in persons:

                if len(person) < 11:
                    continue

                joints = [

                    (person[5], (0,0,255)),
                    (person[6], (0,0,255)),

                    (person[7], (255,0,0)),
                    (person[8], (255,0,0)),

                    (person[9], (0,255,255)),
                    (person[10], (0,255,255))

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