import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from datetime import datetime
import os

# ===============================
#  Load Model
# ===============================
MODEL_PATH = "myopia_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
#  Save Folder
# ===============================
SAVE_FOLDER = "saved_eyes"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ===============================
#  Eye Detector
# ===============================
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ===============================
#  Page Title
# ===============================
st.set_page_config(page_title="Myopia Detector", layout="wide")
st.title("👁️ AI-Based Myopia Detector (Normal / Myopia / Other)")

# ===============================
#  Webcam Input
# ===============================
frame_window = st.image([])
last_eye_frame = None

run = st.checkbox("Start Webcam")

# ===============================
#  Prediction Function
# ===============================
def preprocess_eye(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_eye(eye_img):
    pred = model.predict(eye_img)[0][0]

    if pred < 0.4:
        return "Normal Eye", 0, pred

    elif pred > 0.6:
        confidence = pred * 100
        diopter = -round(confidence / 50, 2)
        return "Myopia Detected", diopter, pred

    else:
        return "Other Eye Condition", None, pred

# ===============================
#  Webcam Loop
# ===============================
if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not accessible")
            break

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        if len(eyes) > 0:
            # Choose largest eye
            x, y, w, h = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[0]
            eye_roi = frame[y:y+h, x:x+w]
            last_eye_frame = eye_roi.copy()

            # Draw bounding box
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        frame_window.image(frame, channels="BGR")

    cap.release()

# ===============================
#  Analyze Button
# ===============================
st.subheader("🔍 Analyze Detected Eye")

if st.button("Analyze Eye Image"):
    if last_eye_frame is None:
        st.warning("No eye detected yet. Turn on webcam first.")
    else:
        processed = preprocess_eye(last_eye_frame)
        label, diopter, raw_pred = predict_eye(processed)

        st.write("### Prediction Result")

        if label == "Normal Eye":
            st.success("🟢 Normal Eye")
            st.write("**Diopter:** 0.00 D")

        elif label == "Myopia Detected":
            st.error("🔴 Myopia Detected")
            st.write(f"**Diopter:** {diopter} D")

        else:
            st.warning("🟡 Other Eye Condition Detected")
            st.write("**Diopter:** -- (Not applicable)")

# ===============================
#  Save Button
# ===============================
st.subheader("💾 Save Eye Image")

if st.button("Save Eye"):
    if last_eye_frame is None:
        st.warning("No image to save.")
    else:
        filename = datetime.now().strftime("eye_%Y%m%d_%H%M%S.jpg")
        path = os.path.join(SAVE_FOLDER, filename)
        cv2.imwrite(path, last_eye_frame)
        st.success(f"Image saved: {path}")
