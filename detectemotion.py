from deepface import DeepFace


def detect_emotions(frame):

    emotions = []

    try:

        results = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )

        if not isinstance(results, list):
            results = [results]

        for result in results:

            region = result["region"]

            x = region["x"]
            y = region["y"]
            w = region["w"]
            h = region["h"]

            emotion = result["dominant_emotion"]

            emotions.append((x, y, w, h, emotion))

    except Exception as e:
        print(e)

    return emotions