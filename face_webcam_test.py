import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import face_recognition
import av

st.title("📷 Face Recognition Webcam Test")

class FaceProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Webcam frame ko RGB mein convert
        rgb = img[:, :, ::-1]

        # Face detection only
        locations = face_recognition.face_locations(
            rgb,
            model="hog"
        )

        # Original frame return
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

webrtc_streamer(
    key="face-test",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)