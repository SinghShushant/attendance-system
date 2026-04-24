import sys
import os
USE_CAMERA = False
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import cv2
from core.detection import detect_faces
from core.preprocessing import preprocess_face
from core.recognition import compare_faces
from core.attendance import mark_attendance
import plotly.express as px

st.set_page_config(page_title="Quantum Attendance", layout="wide")

# 🌌 CSS
st.markdown("""
<style>

/* 🌌 Background with depth */
.stApp {
    background: radial-gradient(circle at 20% 20%, #0f2027, #020c1b 70%);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Hide default */
#MainMenu, footer {visibility: hidden;}

/* ⚡ Neon Title */
.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #00f7ff;
    letter-spacing: 2px;
    text-shadow:
        0 0 10px #00f7ff,
        0 0 20px #00f7ff,
        0 0 40px #00f7ff;
}

/* 🧊 Glass Card (UPGRADED) */
.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(15px);
    box-shadow:
        0 0 10px rgba(0,255,255,0.2),
        0 0 30px rgba(0,255,255,0.1);
    transition: all 0.3s ease;
}

/* Hover glow */
.card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow:
        0 0 20px rgba(0,255,255,0.6),
        0 0 40px rgba(0,255,255,0.3);
}

/* 🧠 Section Title */
.section-title {
    font-size: 26px;
    color: #00ffcc;
    margin-bottom: 10px;
}

/* 📊 Metrics */
.metric {
    font-size: 38px;
    font-weight: bold;
    color: #00ffcc;
}

/* ⚡ Buttons */
.stButton>button {
    background: linear-gradient(45deg, #00f7ff, #00ffcc);
    color: black;
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00f7ff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.8);
}

/* Divider */
hr {
    border: 1px solid #00f7ff;
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
        This system uses <b>Computer Vision</b> to automatically detect and mark student attendance in real-time.
        It eliminates manual attendance, improves accuracy, and ensures efficiency in classrooms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="section-title">🎯 Features</div>
            <ul>
                <li>Real-time face detection</li>
                <li>Automatic attendance logging</li>
                <li>No ML/DL required</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="section-title">⚙️ Technology</div>
            <ul>
                <li>OpenCV (Haar Cascade)</li>
                <li>Histogram Matching</li>
                <li>Edge Detection</li>
                <li>Streamlit UI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="section-title">📊 Benefits</div>
            <ul>
                <li>Reduces manual effort</li>
                <li>Fast & real-time</li>
                <li>Accurate tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="section-title">💡 How It Works</div>
        <p>
        1. Camera captures live video<br>
        2. Faces are detected using Haar Cascade<br>
        3. Recognition is done using histogram + edge matching<br>
        4. Attendance is automatically recorded<br>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================= LIVE =================
elif choice == "Live Attendance":
    st.markdown('<div class="title">🎥 Live Attendance</div>', unsafe_allow_html=True)

    run = st.button("▶ Start Camera")
    stop = st.button("⏹ Stop Camera")
    if not USE_CAMERA:
        st.warning("⚠️ Camera is disabled in deployed version")

    FRAME = st.image([])
    status = st.empty()

    cap = None
    if USE_CAMERA:
        cap = cv2.VideoCapture(0)

    frame_count = {}
    marked_users = set()

    while run and USE_CAMERA:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        faces, gray = detect_faces(frame)
        current_names = []

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = preprocess_face(face)

            name = compare_faces(face)
            current_names.append(name)

            if name != "Unknown":
                frame_count[name] = frame_count.get(name, 0) + 1

                if frame_count[name] == 10 and name not in marked_users:
                    mark_attendance(name)
                    marked_users.add(name)
                    status.success(f"✅ {name} marked present")

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
            cv2.putText(frame, name, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        for name in list(frame_count.keys()):
            if name not in current_names:
                frame_count[name] = 0

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME.image(frame)

        if stop:
            break

    cap.release()

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

        # Chart
        count_df = df["Name"].value_counts().reset_index()
        count_df.columns = ["Name", "Count"]

        fig = px.bar(count_df, x="Name", y="Count", color="Name")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df)

    except:
        st.warning("No data yet")

    if st.button("🧹 Clear Attendance"):
        empty_df = pd.DataFrame(columns=["Name", "Date", "Time"])
        empty_df.to_csv("attendance.csv", index=False)
        st.success("Attendance cleared!")
        st.rerun()

    