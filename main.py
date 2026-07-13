import cv2
import csv
import os

from detectmovement import detect_pose, extract_pose_features, calculate_movement_intensity
from detectemotion import detect_emotions, summarize_emotions_by_boxer, build_emotion_rows


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(BASE_DIR, "video", "2boxers.mp4")

output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "emotion_timeseries.csv")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video.")
    exit()

cv2.namedWindow("Emotion Analyst", cv2.WINDOW_NORMAL)

frame_count = 0
emotions = []
previous_boxers = []

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

last_emotion_second = -1

csv_file = open(csv_path, "w", newline="", encoding="utf-8")

csv_writer = csv.DictWriter(
    csv_file,
    fieldnames=[
        "second",
        "time",
        "boxer_id",
        "emotion",
        "face_x",
        "face_y",
        "face_w",
        "face_h",
        "movement_intensity"
    ]
)

csv_writer.writeheader()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    clean_frame = frame.copy()

    current_second = int(frame_count / fps)

    if current_second != last_emotion_second:
        emotions = detect_emotions(clean_frame)

        current_boxers = extract_pose_features(clean_frame)

        movement_intensities = calculate_movement_intensity(
            current_boxers,
            previous_boxers
        )

        print(summarize_emotions_by_boxer(emotions, current_second))

        rows = build_emotion_rows(emotions, current_second)

        for row in rows:
            boxer_id = row["boxer_id"]

            if boxer_id != "" and boxer_id <= len(movement_intensities):
                row["movement_intensity"] = movement_intensities[boxer_id - 1]
            else:
                row["movement_intensity"] = ""

        csv_writer.writerows(rows)
        csv_file.flush()

        previous_boxers = current_boxers
        last_emotion_second = current_second

    frame = detect_pose(frame)

    sorted_emotions = sorted(emotions, key=lambda detected_emotion: detected_emotion[0])

    for boxer_index, (x, y, w, h, emotion) in enumerate(sorted_emotions, start=1):
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"boxer {boxer_index}: {emotion}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    cv2.imshow("Emotion Analyst", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()

print(f"Emotion time-series saved to: {csv_path}")

cv2.destroyAllWindows()
