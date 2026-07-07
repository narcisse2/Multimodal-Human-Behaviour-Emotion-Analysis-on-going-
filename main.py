import cv2
import os

from detectmovement import detect_pose
from detectemotion import detect_emotions


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(

    BASE_DIR,

    "video",

    "2boxers.mp4"

)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("Cannot open video.")

    exit()

cv2.namedWindow(

    "Emotion Analyst",

    cv2.WINDOW_NORMAL

)

frame_count = 0

emotions = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Pose
    frame = detect_pose(frame)

    # Emotion every 5 frames
    if frame_count % 5 == 0:

        emotions = detect_emotions(frame)

    # Draw emotions
    for (x,y,w,h,emotion) in emotions:

        cv2.rectangle(

            frame,

            (x,y),

            (x+w,y+h),

            (255,0,0),

            2

        )

        cv2.putText(

            frame,

            emotion,

            (x,y-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255,0,0),

            2

        )

    cv2.imshow(

        "Emotion Analyst",

        frame

    )

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()