import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import face_recognition
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import numpy as np
import av
import threading

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="AI Smart Attendance",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Based Smart Attendance Monitoring System")
st.write("Face Recognition Based Automated Attendance System")

# -----------------------------
# FILES
# -----------------------------

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# -----------------------------
# LOAD REGISTERED FACES
# -----------------------------

known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(KNOWN_FACES_DIR, filename)

        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(
                    os.path.splitext(filename)[0]
                )

        except Exception:
            pass

# -----------------------------
# REGISTERED STUDENTS
# -----------------------------

st.subheader("👨‍🎓 Registered Students")

if known_names:

    for name in known_names:
        st.write("✅", name)

else:

    st.warning("No registered students yet.")

# -----------------------------
# REGISTER NEW STUDENT
# -----------------------------

st.subheader("📝 Register New Student")

student_name = st.text_input(
    "Enter Student Name"
)

student_photo = st.file_uploader(
    "Upload Student Photo for Registration",
    type=["jpg", "jpeg", "png"],
    key="registration"
)

if st.button("➕ Register Student"):

    if not student_name:
        st.error("Please enter student name.")

    elif student_photo is None:
        st.error("Please upload a student photo.")

    else:

        image = Image.open(student_photo)
        image_array = np.array(image)

        encodings = face_recognition.face_encodings(
            image_array
        )

        if len(encodings) == 0:

            st.error("❌ No face detected in this photo.")

        elif len(encodings) > 1:

            st.error(
                "❌ Please upload a photo containing only one face."
            )

        else:

            filename = (
                student_name.strip().replace(" ", "_")
                + ".jpg"
            )

            save_path = os.path.join(
                KNOWN_FACES_DIR,
                filename
            )

            image.convert("RGB").save(save_path)

            st.success(
                f"✅ {student_name} registered successfully!"
            )

            st.info(
                "Please refresh the page to load the new student."
            )

# -----------------------------
# LIVE WEBCAM
# -----------------------------

st.subheader("📷 Live Webcam Attendance")

st.write(
    "Click START and allow camera permission. "
    "Face the camera clearly."
)

# -----------------------------
# ATTENDANCE FUNCTION
# -----------------------------

attendance_lock = threading.Lock()


def mark_attendance(name):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    with attendance_lock:

        if os.path.exists(ATTENDANCE_FILE):

            df = pd.read_csv(ATTENDANCE_FILE)

        else:

            df = pd.DataFrame(
                columns=["Name", "Date", "Time"]
            )

        # Already marked today?
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

        df = pd.concat(
            [df, new_row],
            ignore_index=True
        )

        df.to_csv(
            ATTENDANCE_FILE,
            index=False
        )

        return True


# -----------------------------
# WEBCAM PROCESSOR
# -----------------------------

class FaceRecognitionProcessor:

    def __init__(self):

        self.frame_count = 0
        self.last_name = ""
        self.message = ""

    def recv(self, frame):

        self.frame_count += 1

        img = frame.to_ndarray(format="bgr24")

        # Process every 5th frame
        if self.frame_count % 5 == 0:

            rgb = img[:, :, ::-1]

            locations = face_recognition.face_locations(
                rgb,
                model="hog"
            )

            encodings = face_recognition.face_encodings(
                rgb,
                locations
            )

            for encoding, location in zip(
                encodings,
                locations
            ):

                if len(known_encodings) == 0:
                    continue

                distances = face_recognition.face_distance(
                    known_encodings,
                    encoding
                )

                best_match = np.argmin(distances)

                if distances[best_match] < 0.50:

                    name = known_names[best_match]

                    marked = mark_attendance(name)

                    if marked:

                        self.last_name = name
                        self.message = (
                            f"Attendance marked: {name}"
                        )

                    else:

                        self.last_name = name
                        self.message = (
                            f"{name} already marked today"
                        )

                else:

                    self.last_name = ""
                    self.message = "Unknown person"

            # Draw face boxes
            for top, right, bottom, left in locations:

                cv2_color = (0, 255, 0)

                import cv2

                cv2.rectangle(
                    img,
                    (left, top),
                    (right, bottom),
                    cv2_color,
                    2
                )

        # Display name
        if self.message:

            import cv2

            cv2.putText(
                img,
                self.message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# -----------------------------
# START WEBCAM
# -----------------------------

ctx = webrtc_streamer(
    key="face-attendance",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceRecognitionProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    rtc_configuration={
        "iceServers": []
    },
    async_processing=True
)

# -----------------------------
# ATTENDANCE DASHBOARD
# -----------------------------

st.subheader("📊 Attendance Dashboard")

if os.path.exists(ATTENDANCE_FILE):

    dashboard_df = pd.read_csv(
        ATTENDANCE_FILE
    )

    total_students = len(known_names)

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    today_df = dashboard_df[
        dashboard_df["Date"] == today
    ]

    present_today = today_df[
        "Name"
    ].nunique()

    if total_students > 0:

        percentage = (
            present_today /
            total_students
        ) * 100

    else:

        percentage = 0

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👨‍🎓 Total Students",
            total_students
        )

    with col2:

        st.metric(
            "✅ Present Today",
            present_today
        )

    with col3:

        st.metric(
            "📊 Attendance %",
            f"{percentage:.1f}%"
        )

    st.divider()

    st.write("### 📋 Today's Attendance")

    st.dataframe(
        today_df,
        use_container_width=True
    )

else:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "👨‍🎓 Total Students",
            len(known_names)
        )

    with col2:
        st.metric(
            "✅ Present Today",
            0
        )

    with col3:
        st.metric(
            "📊 Attendance %",
            "0.0%"
        )

    st.info(
        "No attendance marked yet."
    )