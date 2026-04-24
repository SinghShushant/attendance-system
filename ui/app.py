
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import cv2
import plotly.express as px
import av

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from core.detection import detect_faces
from core.preprocessing import preprocess_face
from core.recognition import compare_faces
from core.attendance import mark_attendance

st.set_page_config(page_title="Quantum Attendance", layout="wide")

# ================= CSS =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 20% 20%, #0f2027, #020c1b 70%);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
#MainMenu, footer {visibility: hidden;}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #00f7ff;
    text-shadow: 0 0 20px #00f7ff, 0 0 40px #00f7ff;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(15px);
    box-shadow: 0 0 15px rgba(0,255,255,0.2);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 0 25px rgba(0,255,255,0.5);
}

.section-title {
    font-size: 26px;
    color: #00ffcc;
}

.metric {
    font-size: 38px;
    font-weight: bold;
    color: #00ffcc;
}

.stButton>button {
    background: linear-gradient(45deg, #00f7ff, #00ffcc);
    color: black;
    border-radius: 10px;
    font-weight: bold;
}

section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.8);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚡ Quantum Attendance System</div>', unsafe_allow_html=True)

menu = ["Home", "Live Attendance", "Dashboard"]
choice = st.sidebar.selectbox("Menu", menu)

# ================= HOME =================
if choice == "Home":
    st.markdown("""
    <div class="card">
        <h2>🚀 Intelligent Attendance System</h2>
        <p>
        This system uses Computer Vision to automatically detect and mark attendance.
        It removes manual effort and ensures real-time tracking.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="section-title">🎯 Features</div>
            <ul>
                <li>Real-time detection</li>
                <li>Automatic attendance</li>
                <li>No ML required</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="section-title">⚙️ Tech</div>
            <ul>
                <li>OpenCV</li>
                <li>Histogram Matching</li>
                <li>Edge Detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="section-title">📊 Benefits</div>
            <ul>
                <li>Fast</li>
                <li>Accurate</li>
                <li>Automated</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ================= LIVE =================
elif choice == "Live Attendance":
    st.markdown('<div class="title">🎥 Live Attendance</div>', unsafe_allow_html=True)

    status = st.empty()
    frame_count = {}
    marked_users = set()

    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            faces, _ = detect_faces(img)
            current_names = []

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = preprocess_face(face)

                name = compare_faces(face)
                current_names.append(name)

                if name != "Unknown":
                    frame_count[name] = frame_count.get(name, 0) + 1

                    if frame_count[name] == 10 and name not in marked_users:
                        mark_attendance(name)
                        marked_users.add(name)
                        status.success(f"✅ {name} marked present")

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)
                cv2.putText(img, name, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # reset counts
            for name in list(frame_count.keys()):
                if name not in current_names:
                    frame_count[name] = 0

            return img

    webrtc_streamer(
        key="attendance",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# ================= DASHBOARD =================
elif choice == "Dashboard":
    st.markdown('<div class="title">📊 Dashboard</div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        df = pd.read_csv("attendance.csv")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f'<div class="card"><p>Total</p><div class="metric">{len(df)}</div></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="card"><p>Students</p><div class="metric">{df["Name"].nunique()}</div></div>', unsafe_allow_html=True)

        count_df = df["Name"].value_counts().reset_index()
        count_df.columns = ["Name", "Count"]

        fig = px.bar(count_df, x="Name", y="Count", color="Name")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df)

    except:
        st.warning("No attendance data yet")

    if st.button("🧹 Clear Attendance"):
        empty_df = pd.DataFrame(columns=["Name","Date","Time"])
        empty_df.to_csv("attendance.csv", index=False)
        st.success("Attendance cleared!")
        st.rerun()

