import torch

import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

def detect_and_estimate(image):
    # YOLOv5 detection
    results = model(image)

    # MediaPipe pose estimation
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mediapipe = holistic.process(image_rgb)

    # Draw YOLOv5 results
    for result in results.xyxy[0]:  # xyxy is the format for bounding box coordinates
        x1, y1, x2, y2, conf, cls = result
        if int(cls) == 0:  # 0 is the class for 'person' in COCO dataset
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Draw MediaPipe results
    mp.solutions.drawing_utils.draw_landmarks(image, results_mediapipe.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    return image

cap = cv2.VideoCapture(0)  # Capture from webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = detect_and_estimate(frame)
    cv2.imshow('Output', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
