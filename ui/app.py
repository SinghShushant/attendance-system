
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import cv2
import plotly.express as px



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
.card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(0,255,255,0.03));
    border: 1px solid rgba(0,255,255,0.2);
}

.metric {
    font-size: 40px;
    text-shadow: 0 0 10px #00ffcc;
}

.section-title {
    font-weight: bold;
    letter-spacing: 1px;
}            

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚡ Quantum Attendance System</div>', unsafe_allow_html=True)

menu = ["Home", "Add Student", "Live Attendance", "Dashboard"]
choice = st.sidebar.selectbox("Menu", menu)

# ================= HOME =================
if choice == "Home":

    # 🔥 HERO SECTION
    st.markdown("""
    <div class="card" style="text-align:center;">
        <h2>🚀 Intelligent Face Recognition Attendance</h2>
        <p style="font-size:18px;">
        A next-generation attendance system powered by Computer Vision.
        Automatically detects faces, identifies individuals, and records attendance in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 📊 SYSTEM STATS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="card">
            <p>📸 Faces Detected</p>
            <div class="metric">Real-time</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <p>⚡ Speed</p>
            <div class="metric">Fast</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <p>🎯 Accuracy</p>
            <div class="metric">High</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card">
            <p>🔒 Security</p>
            <div class="metric">Secure</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 🧠 FEATURES SECTION
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="section-title">🎯 Features</div>
            <ul>
                <li>Real-time face detection</li>
                <li>Automatic attendance logging</li>
                <li>Multiple user support</li>
                <li>No manual input required</li>
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
                <li>Streamlit Interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="section-title">📊 Benefits</div>
            <ul>
                <li>Reduces manual effort</li>
                <li>Improves accuracy</li>
                <li>Time efficient</li>
                <li>Easy deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 🔥 HOW IT WORKS
    st.markdown("""
    <div class="card">
        <div class="section-title">💡 How It Works</div>
        <p>
        1️⃣ Camera captures live video<br><br>
        2️⃣ Faces are detected using Haar Cascade<br><br>
        3️⃣ Recognition is done using histogram + edge matching<br><br>
        4️⃣ Attendance is automatically recorded into database<br><br>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ⚡ STATUS PANEL (NEW)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="section-title">🟢 System Status</div>
            <p>Camera: Ready</p>
            <p>Model: Loaded</p>
            <p>Database: Connected</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="section-title">🚀 Use Cases</div>
            <p>✔ College Attendance</p>
            <p>✔ Office Tracking</p>
            <p>✔ Smart Classrooms</p>
        </div>
        """, unsafe_allow_html=True)

# ================= Student UI =================

elif choice == "Add Student":
    st.markdown('<div class="title">➕ Add New Student</div>', unsafe_allow_html=True)

    import os
    import time

    name = st.text_input("Enter Student Name")

    start = st.button("📸 Start Capture")

    if name and start:
        dataset_path = f"dataset/{name}"

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        cap = cv2.VideoCapture(0)

        st.info("Look at camera... Capturing will start")

        count = 0
        FRAME_WINDOW = st.image([])

        while count < 20:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not working")
                break

            # Save image
            img_path = f"{dataset_path}/{count}.jpg"
            cv2.imwrite(img_path, frame)

            # Show live
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

            count += 1
            time.sleep(0.2)

        cap.release()
        st.success(f"✅ {name} added successfully!")

# ================= LIVE =================

elif choice == "Live Attendance":
    st.markdown('<div class="title">🎥 Live Attendance</div>', unsafe_allow_html=True)

    # 🔁 CHANGE THIS BASED ON WHERE YOU RUN
    USE_CAMERA = True   # 👉 True for local, False for deployed

    if not USE_CAMERA:
        st.warning("⚠️ Camera not supported in deployed version")
        st.info("Run this project locally to use live attendance.")
    else:
        start = st.button("▶ Start Camera")
        stop = st.button("⏹ Stop Camera")

        FRAME_WINDOW = st.image([])
        status = st.empty()

        frame_count = {}
        marked_users = set()

        if start:
            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera not working")
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

                        # ✅ Only once per session
                        if frame_count[name] == 10 and name not in marked_users:
                            mark_attendance(name)
                            marked_users.add(name)
                            status.success(f"✅ {name} marked present")

                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                    cv2.putText(frame, name, (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                # Reset logic
                for name in list(frame_count.keys()):
                    if name not in current_names:
                        frame_count[name] = 0

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

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

