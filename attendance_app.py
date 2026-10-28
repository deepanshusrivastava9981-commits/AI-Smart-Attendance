import streamlit as st
import face_recognition
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="AI Smart Attendance",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Based Smart Attendance Monitoring System")
st.write("Face Recognition Based Automated Attendance System")

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


# ---------------- LOAD REGISTERED STUDENTS ----------------

def load_known_faces():
    encodings = []
    names = []

    for filename in os.listdir(KNOWN_FACES_DIR):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            path = os.path.join(KNOWN_FACES_DIR, filename)

            try:
                image = face_recognition.load_image_file(path)
                face_encodings = face_recognition.face_encodings(image)

                if len(face_encodings) > 0:
                    encodings.append(face_encodings[0])
                    names.append(os.path.splitext(filename)[0])

            except Exception:
                pass

    return encodings, names


known_encodings, known_names = load_known_faces()


# ---------------- REGISTER STUDENT ----------------

st.subheader("👨‍🎓 Registered Students")

if known_names:

    for name in known_names:
        st.write("✅", name)

else:
    st.info("No students registered yet.")


st.subheader("📝 Register New Student")

student_name = st.text_input("Enter Student Name")

student_photo = st.file_uploader(
    "Upload Student Photo",
    type=["jpg", "jpeg", "png"],
    key="register_photo"
)

if st.button("➕ Register Student"):

    if not student_name.strip():

        st.error("Please enter student name.")

    elif student_photo is None:

        st.error("Please upload a student photo.")

    else:

        image = Image.open(student_photo).convert("RGB")
        image_array = np.array(image)

        encodings = face_recognition.face_encodings(image_array)

        if len(encodings) == 0:

            st.error("❌ No face detected.")

        elif len(encodings) > 1:

            st.error("❌ Please upload a photo with only one face.")

        else:

            filename = (
                student_name.strip().replace(" ", "_")
                + ".jpg"
            )

            save_path = os.path.join(
                KNOWN_FACES_DIR,
                filename
            )

            image.save(save_path)

            st.success(
                f"✅ {student_name} registered successfully!"
            )

            st.info("Refresh the page to load the new student.")


# ---------------- ATTENDANCE ----------------

st.subheader("📷 Webcam Attendance")

st.write(
    "Click **Take Photo**, show your face clearly, "
    "and attendance will be marked automatically."
)

camera_photo = st.camera_input(
    "📸 Take Attendance Photo"
)

if camera_photo is not None:
    st.success("📸 Photo received successfully!")



# ---------------- MARK ATTENDANCE ----------------

def mark_attendance(name):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):

        df = pd.read_csv(ATTENDANCE_FILE)

    else:

        df = pd.DataFrame(
            columns=["Name", "Date", "Time"]
        )

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


# ---------------- FACE RECOGNITION ----------------

if camera_photo is not None:

    image = Image.open(camera_photo).convert("RGB")
    image_array = np.array(image)

    locations = face_recognition.face_locations(
        image_array,
        model="hog"
    )
    st.write("🔍 Faces detected:", len(locations))

    encodings = face_recognition.face_encodings(
        image_array,
        locations
    )

    if len(encodings) == 0:

        st.error("❌ No face detected. Please try again.")

    else:

        recognized = False

        for encoding in encodings:

            if len(known_encodings) == 0:
                continue

            distances = face_recognition.face_distance(
                known_encodings,
                encoding
            )

            best_match = np.argmin(distances)
            st.write("🔍 Face distance:", float(distances[best_match]))

            if distances[best_match] < 0.50:
                    name = known_names[best_match]
                    st.success(f"🎯 Face Recognized: {name}")

                    marked = mark_attendance(name)

                    if marked:
                        st.success(
                            f"✅ Attendance marked successfully: {name}"
                        )
                    else:
                        st.info(
                            f"ℹ️ {name} is already marked today."
                        )

                    recognized = True