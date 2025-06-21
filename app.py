from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

import numpy as np
from feat import Detector

emotion_au_map = {
    'Happiness': {6, 12},
    'Sadness':   {1, 4, 15},
    'Surprise':  {1, 2, 5, 26},
    'Fear':      {1, 2, 4, 5, 7, 20, 26},
    'Anger':     {4, 5, 7, 23},
    'Disgust':   {9, 15},
    'Contempt':  {12, 14},
}

AU_THRESH = 0.5

au_cols = [
    "AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10",
    "AU11","AU12","AU14","AU15","AU17","AU20","AU23","AU24",
    "AU25","AU26","AU28","AU43",
]
au_idx_to_num = {i: int(c[2:]) for i, c in enumerate(au_cols)}

detector = Detector(device='cpu')

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
        
    faces_out = detector.detect_faces(img)
    
    if not faces_out or len(faces_out) == 0 or len(faces_out[0]) == 0 :
        return jsonify({"error": "No faces detected"}), 404

    faces_arr = faces_out[0]
    x1, y1, x2, y2, _ = faces_arr[0]

    lms_out = detector.detect_landmarks(img, faces_out)
    aus_out = detector.detect_aus(img, lms_out)
    if not aus_out:
        return jsonify({"error": "Got no aus"}), 404

    au_vals = aus_out[0][0]

    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    active = {
        au_idx_to_num[i]
        for i, p in enumerate(au_vals)
        if p > AU_THRESH
    }
    emotion = 'Neutral'
    for emo, req in emotion_au_map.items():
        if req.issubset(active):
            emotion = emo
            break
        
    return jsonify({
        "emotion": emotion,
    })

if __name__ == "__main__":
    app.run(debug=True)