import streamlit as st
import face_recognition
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import numpy as np

# Page settings
st.set_page_config(
    page_title="AI Smart Attendance",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Based Smart Attendance Monitoring System")
st.write("Face Recognition Based Automated Attendance System")

# Files and folders
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

# Create known_faces folder if it doesn't exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load registered faces
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(
            KNOWN_FACES_DIR,
            filename
        )

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])

                name = os.path.splitext(filename)[0]
                known_names.append(name)

        except Exception as e:
            st.warning(f"Could not process {filename}: {e}")


# Registered students
st.subheader("👨‍🎓 Registered Students")

if known_names:
    for name in known_names:
        st.write("✅", name)
else:
    st.warning("No registered student photos found.")


# Upload photo
st.subheader("📷 Mark Attendance")

uploaded_file = st.file_uploader(
    "Upload Student Photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Student Photo",
        width=400
    )

    image_array = np.array(image)

    # Detect faces
    face_locations = face_recognition.face_locations(
        image_array
    )

    face_encodings = face_recognition.face_encodings(
        image_array,
        face_locations
    )

    if len(face_encodings) == 0:

        st.error("❌ No face detected.")

    elif len(known_encodings) == 0:

        st.error("❌ No registered faces found.")

    else:

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=0.5
            )

            distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_match = np.argmin(distances)

            if matches[best_match]:

                name = known_names[best_match]

                now = datetime.now()

                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                # Load attendance file
                if os.path.exists(ATTENDANCE_FILE):

                    df = pd.read_csv(ATTENDANCE_FILE)

                else:

                    df = pd.DataFrame(
                        columns=["Name", "Date", "Time"]
                    )

                # Check today's attendance
                already_marked = (
                    (df["Name"] == name) &
                    (df["Date"] == date)
                ).any()

                if already_marked:

                    st.info(
                        f"ℹ️ {name}'s attendance is already marked today."
                    )

                else:

                    new_data = pd.DataFrame({
                        "Name": [name],
                        "Date": [date],
                        "Time": [time]
                    })

                    df = pd.concat(
                        [df, new_data],
                        ignore_index=True
                    )

                    df.to_csv(
                        ATTENDANCE_FILE,
                        index=False
                    )

                    st.success(
                        f"✅ Attendance marked for {name}!"
                    )

                    st.write(f"📅 Date: {date}")
                    st.write(f"⏰ Time: {time}")

            else:

                st.error("❌ Student not recognized.")


# Attendance records
st.subheader("📋 Attendance Records")

if os.path.exists(ATTENDANCE_FILE):

    df = pd.read_csv(ATTENDANCE_FILE)

    if len(df) > 0:

        st.dataframe(
            df,
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Attendance CSV",
            data=df.to_csv(index=False),
            file_name="attendance.csv",
            mime="text/csv"
        )

    else:

        st.info("No attendance records yet.")

else:

    st.info("No attendance records yet.")