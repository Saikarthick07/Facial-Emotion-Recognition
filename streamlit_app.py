import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the pre-trained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
classifier = load_model('model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    labels = []
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
            labels.append(label)
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame, labels

def main():
    st.title('Facial Emotion Recognition')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        processed_image, labels = detect_emotion(image)
        st.image(processed_image, channels="BGR", caption="Processed Image")
        st.write("Predicted Emotions:", ", ".join(labels))

    else:
        st.subheader("Live Webcam Feed")
        if st.button('Start Webcam'):
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Error: Unable to open webcam.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Warning: Unable to retrieve frame from webcam.")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame, labels = detect_emotion(frame)
                st.image(processed_frame, channels="RGB", caption="Live Webcam Feed")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

if __name__ == '__main__':
    main()
