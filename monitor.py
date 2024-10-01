import streamlit as st
import cv2
import numpy as np
from pyppg.ppg import PPG
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

class HeartRateProcessor(VideoProcessorBase):
    def __init__(self):
        self.ppg = PPG()
        self.ppg_signal = []
        self.start_time = time.time()
        self.analysis_duration = 10
        self.fs = 30  # Assuming 30 fps, adjust if needed

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract PPG signal (using mean pixel value of the face region)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            ppg_value = np.mean(face_roi)
            self.ppg_signal.append(ppg_value)
        
        # Process PPG signal when we have enough data
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.analysis_duration:
            # Convert to numpy array
            ppg_array = np.array(self.ppg_signal)
            
            # Analyze PPG signal
            results = self.ppg.ppg_analysis(ppg_array, fs=self.fs)
            
            # Update heart rate in Streamlit
            st.session_state.heart_rate = f"{results.hr:.2f}"
            
            # Reset for next analysis
            self.ppg_signal = []
            self.start_time = time.time()
        
        return img

def main():
    st.title("Heart Rate Monitor using Webcam")
    
    st.write("This app uses your webcam to estimate your heart rate.")
    st.write("Please ensure you are in a well-lit environment and your face is clearly visible.")
    
    if 'heart_rate' not in st.session_state:
        st.session_state.heart_rate = "Calculating..."
    
    webrtc_streamer(
        key="heart-rate",
        video_processor_factory=HeartRateProcessor,
        async_processing=True,
    )
    
    st.write("Estimated Heart Rate:")
    heart_rate_placeholder = st.empty()
    
    while True:
        heart_rate_placeholder.write(f"{st.session_state.heart_rate} bpm")
        time.sleep(1)

if __name__ == "__main__":
    main()