from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

import numpy as np
from feat import Detector

detector = Detector(device='cpu')

FEAT_EMOTION_COLUMNS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

app = Flask(__name__)

@app.route("/",)
def root():
    return "Server is up"
    
@app.route("/upload", methods=["POST"])
def upload():
    img_bytes = request.data
    if not img_bytes:
        return jsonify({"error": "No image data received"}), 400

    try:
        img = Image.open(BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": "Invalid JPEG data"}), 400
    
    img = img.resize((224, 224), resample=Image.LANCZOS)
    
    try:
        faces_out = detector.detect_faces(img)
        if not faces_out or len(faces_out) == 0 or len(faces_out[0]) == 0 :
            return jsonify({"error": "No faces detected"}), 404
    except Exception as e:
        return jsonify({"error": "No faces detected"}), 404
    
    try:
        detected_landmarks = detector.detect_landmarks(img, faces_out)
    except Exception as e:
        return jsonify({"error": "Error detecting landmark"}), 404
    
    try:
        emotions = detector.detect_emotions(img, faces_out, detected_landmarks)
    except Exception as e:
        return jsonify({"error": "Error detecting emotions"}), 404
    
    pred_label = FEAT_EMOTION_COLUMNS[np.argmax(emotions[0][0])]
    
    return jsonify({
        "emotion": pred_label,
    })

if __name__ == "__main__":
    app.run(debug=True)