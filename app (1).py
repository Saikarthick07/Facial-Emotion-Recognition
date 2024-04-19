import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the pre-trained face detection model
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
classifier = load_model(r'model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to process the image and detect emotions
def detect_emotion(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Create a Streamlit app
st.title('Emotion Detection App')

# Add a section to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display the output feed for uploaded image
if uploaded_file is not None:
    image = detect_emotion(uploaded_file)
    st.image(image, channels="BGR", caption="Processed Image")

# Add a section for live webcam feed
st.subheader("Live Webcam Feed")

# Function to capture webcam feed and detect emotions
def detect_emotions_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = detect_emotion(frame)
        st.image(frame, channels="RGB", caption="Live Webcam Feed")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

# Button to start live webcam feed
if st.button('Start Webcam'):
    detect_emotions_webcam()
