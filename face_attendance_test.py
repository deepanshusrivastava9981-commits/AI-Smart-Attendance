import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import face_recognition
import av
import os
import numpy as np
import pandas as pd
from datetime import datetime
import cv2

st.title("🤖 Face Attendance Test")

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

# Load registered faces
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_FACES_DIR, filename)

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

st.write("Registered students:", known_names)


def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    already_marked = (
        (df["Name"] == name) &
        (df["Date"] == today)
    ).any()

    if already_marked:
        return False

    new_row = pd.DataFrame({
        "Name": [name],
        "Date": [today],
        "Time": [current_time]
    })

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)

    return True


class FaceProcessor:

    def __init__(self):
        self.frame_count = 0
        self.last_message = ""

    def recv(self, frame):

        self.frame_count += 1

        img = frame.to_ndarray(format="bgr24")

        # Process every 10th frame
        if self.frame_count % 10 == 0:

            rgb = img[:, :, ::-1]

            locations = face_recognition.face_locations(
                rgb,
                model="hog"
            )

            encodings = face_recognition.face_encodings(
                rgb,
                locations
            )

            for encoding, location in zip(encodings, locations):

                if len(known_encodings) == 0:
                    continue

                distances = face_recognition.face_distance(
                    known_encodings,
                    encoding
                )

                best_match = np.argmin(distances)

                if distances[best_match] < 0.50:

                    name = known_names[best_match]

                    if mark_attendance(name):
                        self.last_message = f"✅ Attendance marked: {name}"
                    else:
                        self.last_message = f"ℹ️ {name} already marked today"

                else:
                    self.last_message = "❌ Unknown person"

                top, right, bottom, left = location

                cv2.rectangle(
                    img,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

        if self.last_message:

            cv2.putText(
                img,
                self.last_message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="face-attendance-test",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)