import streamlit as st
import cv2
import numpy as np
import time
import pyPPG

# Function to process PPG signal and calculate BPM
def calculate_bpm(signal, fps):
    # Use pyPPG to process the signal
    ppg_processor = pyPPG.PPGProcessor()
    processed_signal = ppg_processor.process(signal)
    
    # Simple peak detection algorithm
    peaks = (processed_signal > np.mean(processed_signal)).astype(int)
    peak_times = np.where(peaks == 1)[0] / fps
    intervals = np.diff(peak_times)
    bpm = 60 / np.mean(intervals)
    return bpm

# Streamlit app
st.title('Real-time Heart Rate Monitoring using pyPPG')
run = st.checkbox('Run')

# Initialize video capture
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize variables
signal = []
timestamps = []

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract mean intensity from a region of interest (ROI)
    roi = gray[100:200, 100:200]
    mean_intensity = np.mean(roi)
    signal.append(mean_intensity)
    timestamps.append(time.time())

    # Display the video frame
    st.image(frame, channels="BGR")

    # Process the signal if we have enough data points
    if len(signal) > fps * 10:  # 10 seconds of data
        bpm = calculate_bpm(np.array(signal[-int(fps*10):]), fps)
        st.write(f"Heart Rate: {bpm:.2f} BPM")

cap.release()
