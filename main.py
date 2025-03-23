import torch
import mediapipe as mp
import streamlit as st
import cv2
import numpy as np

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5s.pt')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Streamlit Dashboard Setup
st.title("YOLOv5 + MediaPipe Pose Estimation with Streamlit")
video_placeholder = st.empty()

# Setup webcam stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 object detection
    results = model(frame)
    for *xyxy, conf, cls in results.xyxy[0]:
        label = f"{model.names[int(cls)]} {conf:.2f}"
        xyxy = list(map(int, xyxy))
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # MediaPipe Pose Estimation
    results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results_pose.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

# Release the video capture object
cap.release()
