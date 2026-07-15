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
#keep the same ids for boxers at the beginning
def assign_stable_boxer_ids(current_boxers, previous_tracks, max_boxers = 2):
    if not current_boxers:
        return [], previous_tracks
    # no previous boxer exist, assign boxer ids left to right
    if not previous_tracks:
        tracked_boxers = []
        new_tracks = {}
        
        for boxer_id, boxer in enumerate(current_boxers[:max_boxers], start=1):
            #copy boxer data so we can add the stable id
            tracked_boxer = boxer.copy()
            tracked_boxer["boxer_id"] = boxer_id 
            tracked_boxers.append(tracked_boxer)
            #save the boxer as the lastest known position for this id
            new_tracks[boxer_id] = tracked_boxer
            
        
        return tracked_boxers, new_tracks 
    # build all possible matches between current detections and previous tracks
    #each candidate store:distance between positions, currentboxer index, previous boxer id
    candidates = []
    
    for current_index, boxer in enumerate(current_boxers[:max_boxers]):
        current_x, current_y = boxer["center"]
        
        for boxer_id , previous_boxer in previous_tracks.items():
            previous_x, previous_y = previous_boxer["center"]
            #small distance mean this boxer probly the same
            distance = ((current_x - previous_x)** 2 + (current_y - previous_y) ** 2) ** 0.5 
            candidates.append((distance, current_index, boxer_id))
    #try closest match first        
    candidates.sort(key=lambda candidate: candidate[0])
    #prevent assigning one detection to multiple ids or 1 id to multiple detections
    assigned_current = set()
    assigned_ids = set()
    tracked_boxers = []
    new_tracks = previous_tracks.copy()
    
    for _, current_index, boxer_id in candidates:
        if current_index in assigned_current or boxer_id in assigned_ids:
            continue
        tracked_boxer = current_boxers[current_index].copy()
        tracked_boxer["boxer_id"] = boxer_id
        tracked_boxers.append(tracked_boxer)
        new_tracks[boxer_id ] = tracked_boxer
        
        #mark detection+ ids as used
        assigned_current.add(current_index)
        assigned_ids.add(boxer_id)
    
    # if a current detection was not matched, give unused boxer id
    #this can happpen if a boxer  was missiing before apprears again
    
    available_ids= [
        boxer_id
        for boxer_id in range(1, max_boxers + 1)
        if boxer_id not in assigned_ids
        
    ]   
    
    for current_index, boxer in enumerate(current_boxers[:max_boxers]):
        if current_index in assigned_current or not available_ids:
            continue
        
        boxer_id = available_ids.pop(0)
        tracked_boxer = boxer.copy()
        tracked_boxer["boxer_id"] = boxer_id
        tracked_boxers.append(tracked_boxer)
        new_tracks[boxer_id] = tracked_boxer
        
        
    tracked_boxer.sort(key=lambda boxer: boxer["boxer_id"])
    
    return tracked_boxers, new_tracks


def calculate_movement_by_id(current_tracks, previous_tracks):
    movements = {}
    
    for boxer_id, current_boxer in current_tracks.items():
        # if boxer new , no previous position to compare with
        if boxer_id  not in previous_tracks:
            movements[boxer_id] = 0.0
            continue
        #
        #get current and previous body center
        current_x, current_y = current_boxer["center"]
        previous_x, previous_y = previous_tracks[boxer_id]["center"]
        
        #different in horizontal and verical movement
        
        dx = current_x - previous_x
        dy = current_y - previous_y
        
        


