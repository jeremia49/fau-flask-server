from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image

import numpy as np
from deepface import DeepFace

def to_serializable(obj):
    """
    Recursively convert NumPy types to Python built-ins:
      - np.ndarray → list
      - np.generic   → native scalar (.item())
      - dict, list   → recurse
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


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
    
    try:
        objs = DeepFace.analyze(
            img_path = np.array(img), actions = ['emotion'],
        )
    except Exception as e:
        print(e)
        return jsonify({"error": "No face detected"}), 404
    
    if(len(objs) == 0):
        return jsonify({"error": "No face detected"}), 404
    
    return jsonify({
        "emotion": objs[0]['dominant_emotion'],
        "stats": to_serializable(objs[0]['emotion'])
    })

if __name__ == "__main__":
    app.run(debug=True)