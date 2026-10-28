import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("📷 Webcam Test")

webrtc_streamer(
    key="webcam",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)