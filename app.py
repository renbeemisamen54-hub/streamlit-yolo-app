import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# Define the RTC configuration for STUN servers (Critical for Cloud)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Cache the model so it doesn't reload every rerun
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Video frame callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    # Use persist=True to keep track of objects across frames
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # Annotate frame
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Start WebRTC streamer
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV, # Explicitly set mode
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
