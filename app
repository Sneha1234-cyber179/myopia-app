import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# ====== Load Trained Model ======
MODEL_PATH = r"C:\Users\Sneha.G\Documents\myopia_detector_app\myopia_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ====== Save Folder ======
SAVE_FOLDER = r"C:\Users\Sneha.G\Downloads\MyopiaDetectionSave"

# Create folder if not exists
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# ====== Eye Detector ======
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ====== Tkinter Window ======
root = Tk()
root.title("Eye Diopter Scanner (Enhanced)")
root.geometry("900x750")
root.configure(bg="#1e2a38")

# ====== Title ======
Label(root, text="👁️ AI-Based Myopia Detector (Enhanced)",
      font=("Helvetica", 20, "bold"),
      bg="#1e2a38", fg="white").pack(pady=15)

# ====== Webcam Display ======
video_frame = Label(root, bg="#1e2a38")
video_frame.pack()

# ====== Result Area ======
result_frame = Frame(root, bg="#2f3e52", bd=2, relief=RIDGE)
result_frame.pack(fill="x", padx=40, pady=20, ipady=10)

prediction_lbl = Label(result_frame, text="Prediction:",
                       font=("Arial", 14, "bold"), bg="#2f3e52", fg="white")
prediction_lbl.grid(row=0, column=0, padx=10, pady=5, sticky=W)

prediction_val = Label(result_frame, text="--",
                       font=("Arial", 14, "bold"), bg="#2f3e52", fg="yellow")
prediction_val.grid(row=0, column=1, padx=10, pady=5, sticky=W)

diopter_lbl = Label(result_frame, text="Diopter:",
                    font=("Arial", 14, "bold"), bg="#2f3e52", fg="white")
diopter_lbl.grid(row=1, column=0, padx=10, pady=5, sticky=W)

diopter_val = Label(result_frame, text="--",
                    font=("Arial", 14, "bold"), bg="#2f3e52", fg="yellow")
diopter_val.grid(row=1, column=1, padx=10, pady=5, sticky=W)

# ====== Webcam ======
cap = cv2.VideoCapture(0)

def preprocess_eye(eye_img):
    img = cv2.resize(eye_img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

last_eye_frame = None  # To store last detected eye image


# ====== ANALYZE FRAME ======
def analyze_frame():
    global last_eye_frame

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Webcam not accessible.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    if len(eyes) == 0:
        prediction_val.config(text="NO EYE DETECTED", fg="orange")
        diopter_val.config(text="--", fg="orange")
        return

    # Take largest detected eye
    x, y, w, h = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[0]
    eye_roi = frame[y:y + h, x:x + w]
    last_eye_frame = eye_roi.copy()

    input_img = preprocess_eye(eye_roi)
    pred = model.predict(input_img)[0][0]

    # ===== THREE-CLASS LOGIC FOR OTHER =====
    if pred < 0.4:
        prediction_val.config(text="NORMAL EYE", fg="lightgreen")
        diopter_val.config(text="0 D", fg="lightgreen")

    elif 0.4 <= pred <= 0.6:
        prediction_val.config(text="OTHER / UNKNOWN PATTERN", fg="yellow")
        diopter_val.config(text="--", fg="yellow")

    else:
        confidence = pred * 100
        diopter = -round(confidence / 50, 2)
        prediction_val.config(text=f"MYOPIA DETECTED ({confidence:.1f}%)", fg="red")
        diopter_val.config(text=f"{diopter} D", fg="red")


# ====== SAVE IMAGE BUTTON ======
def save_eye_image():
    global last_eye_frame

    if last_eye_frame is None:
        messagebox.showinfo("Info", "No eye image to save. Click Analyze first.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(SAVE_FOLDER, f"eye_{timestamp}.jpg")

    cv2.imwrite(file_path, last_eye_frame)
    messagebox.showinfo("Saved", f"Image saved:\n{file_path}")


# ====== Live Video ======
def update_video():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        video_frame.imgtk = imgtk
        video_frame.configure(image=imgtk)

    video_frame.after(10, update_video)


# ====== Buttons ======
Button(root, text="Analyze Current Frame",
       command=analyze_frame,
       bg="#1abc9c", fg="white",
       font=("Arial", 14, "bold"), padx=20, pady=10).pack(pady=10)

Button(root, text="Save Eye Image",
       command=save_eye_image,
       bg="#3498db", fg="white",
       font=("Arial", 14, "bold"), padx=20, pady=10).pack(pady=5)

# ====== Start Webcam ======
update_video()

# ====== Exit ======
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
