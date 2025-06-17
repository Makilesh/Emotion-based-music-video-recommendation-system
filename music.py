import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import webbrowser
import os

# Check if model and label files exist
model_path = "new_model.h5"
labels_path = "labels.npy"

if not os.path.exists(model_path):
    st.error("Model file not found.")
else:
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None

if not os.path.exists(labels_path):
    st.error("Labels file not found.")
else:
    try:
        label = np.load(labels_path)
        st.success("Labels loaded successfully!")
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        label = None

# Initialize MediaPipe holistic model
holistic = mp.solutions.holistic
drawing = mp.solutions.drawing_utils

st.header("Emotion-Based Music Recommender")

# Set up session state for controlling webcam functionality
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load previously saved emotion if available
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = None

# Emotion Processor class to handle webcam feed
class EmotionProcessor:
    def __init__(self):
        self.holis = holistic.Holistic()

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)  # Flip the frame horizontally
        try:
            res = self.holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            if res.face_landmarks or res.left_hand_landmarks or res.right_hand_landmarks:
                # Draw face and hand landmarks if detected
                if res.face_landmarks:
                    drawing.draw_landmarks(
                        frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                        connection_drawing_spec=drawing.DrawingSpec(thickness=1)
                    )
                if res.left_hand_landmarks:
                    drawing.draw_landmarks(frm, res.left_hand_landmarks, holistic.HAND_CONNECTIONS)
                if res.right_hand_landmarks:
                    drawing.draw_landmarks(frm, res.right_hand_landmarks, holistic.HAND_CONNECTIONS)
            else:
                print("No landmarks detected.")
        except Exception as e:
            print(f"Error processing frame: {e}")

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

    def __del__(self):
        self.holis.close()


# Collect user inputs for language and singer preferences
lang = st.text_input("Language")
singer = st.text_input("Singer")

# Initiate webcam feed only if lang and singer inputs are provided
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(
    key="emotion_processor",
    desired_playing_state=True,
    video_processor_factory=EmotionProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )



# Button to trigger song recommendation based on detected emotion
btn = st.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        # Open YouTube with a search query combining language, detected emotion, and singer
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        np.save("emotion.npy", np.array([""]))  # Reset emotion after recommendation
        st.session_state["run"] = "false"
