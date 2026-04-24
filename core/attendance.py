import pandas as pd
from datetime import datetime
import os
import threading

FILE = "attendance.csv"

def speak(text):
    os.system(f"say '{text}'")

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if not os.path.exists(FILE):
        df = pd.DataFrame(columns=["Name","Date","Time"])
        df.to_csv(FILE, index=False)

    df = pd.read_csv(FILE)

    # allow multiple entries (for demo)
    df.loc[len(df)] = [name, date, time]
    df.to_csv(FILE, index=False)

    print(f"{name} marked")

    # 🔊 sound
    threading.Thread(target=speak, args=(f"{name} marked present",)).start()