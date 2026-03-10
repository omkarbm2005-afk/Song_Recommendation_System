from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from fer import FER

app = Flask(__name__)
detector = FER()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    result = detector.detect_emotions(frame)

    if result:
        emotions = result[0]["emotions"]
        emotion = max(emotions, key=emotions.get)
    else:
        emotion = "neutral"

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)