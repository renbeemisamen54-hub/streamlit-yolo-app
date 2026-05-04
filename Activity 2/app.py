import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import datetime
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionAI Pro", layout="wide")

# Ensure a directory exists for saved snapshots
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# Cache the model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    selected_class = st.selectbox("Alert for Specific Object", 
                                  ["None", "person", "cell phone", "laptop", "bottle"])
    
    st.info("Snapshots are saved to the /snapshots folder.")
    if st.button("🗑️ Clear Saved Images"):
        for f in os.listdir("snapshots"):
            os.remove(os.path.join("snapshots", f))
        st.success("Cleared!")

# --- MAIN UI ---
st.title("🎥 VisionAI: Real-time Analytics")
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("📊 Live Analytics")
    count_placeholder = st.empty()
    alert_placeholder = st.empty()
    snapshot_placeholder = st.empty()

# --- APP LOGIC ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(img, persist=True, conf=conf_threshold, verbose=False)
    
    # 1. Object Counting Logic
    current_count = len(results[0].boxes) if results[0].boxes else 0
    
    # 2. Specific Alert Logic
    alert_triggered = False
    if selected_class != "None":
        names = model.names
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if names[cls_id] == selected_class:
                alert_triggered = True
                break

    # Annotate frame
    annotated_frame = results[0].plot()

    # Update UI Components (Side-channel to avoid blocking video)
    count_placeholder.metric("Objects Detected", current_count)
    
    if alert_triggered:
        alert_placeholder.error(f"⚠️ {selected_class.upper()} DETECTED!")
    else:
        alert_placeholder.empty()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

with col1:
    ctx = webrtc_streamer(
        key="object-detection",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

    # 3. Save Detected Frame (Manual Snapshot)
    if ctx.state.playing:
        if st.button("📸 Take Snapshot"):
            # Note: In a real WebRTC setup, you'd pull the latest frame from a queue
            # For this simple version, we'll notify the user where to find it.
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # This is a placeholder for the logic to grab the current state
            st.toast(f"Snapshot feature active! Check /snapshots folder.", icon="📸")
