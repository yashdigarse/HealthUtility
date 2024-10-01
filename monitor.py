# webcam_heart_rate.py

import cv2
import streamlit as st
import numpy as np
from pyppg import PPG

def main():
    st.title("Webcam Heart Rate Monitor")

    cap = cv2.VideoCapture(0)  # Open the default camera (webcam)

    ppg = PPG()  # Initialize pyPPG

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame (you can add more preprocessing here)
        # For demonstration, let's just display the frame
        st.image(frame, channels="BGR", use_column_width=True)

        # Extract PPG signal (you'll need to adjust this based on your ROI)
        # ppg_signal = extract_ppg_signal(frame)

        # Calculate heart rate using pyPPG
        # heart_rate = ppg.process(ppg_signal)

        # Display heart rate (uncomment when ready)
        # st.write(f"Heart Rate: {heart_rate} BPM")

    cap.release()

if __name__ == "__main__":
    main()
