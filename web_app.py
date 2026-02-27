from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

# Load model assets once at startup
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifier = load_model("model.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")

    if not image_data.startswith("data:image"):
        return jsonify({"error": "Invalid image payload."}), 400

    try:
        image_b64 = image_data.split(",", maxsplit=1)[1]
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({"emotion": "No face detected"})

        largest_face = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = largest_face
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        roi = roi_gray.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi, verbose=0)[0]
        label = emotion_labels[int(prediction.argmax())]
        confidence = float(np.max(prediction))

        return jsonify({"emotion": label, "confidence": round(confidence, 3)})
    except Exception as exc:  # defensive for malformed camera payloads
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
