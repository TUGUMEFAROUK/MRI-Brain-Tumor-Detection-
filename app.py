import tensorflow as tf
import numpy as np
from PIL import Image

from flask import Flask, render_template, request


# -------------------------
# LOAD TRAINED MODEL
# -------------------------

model = tf.keras.models.load_model(
    "brain_tumor_model.keras",
    compile=False
)

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

print("Model loaded successfully!")


# -------------------------
# PREDICTION FUNCTION
# -------------------------

def predict_image(image_path):

    # Load and resize exactly as during training
    image = Image.open(image_path).convert("RGB").resize((128, 128))
    image = np.asarray(image, dtype=np.float32)

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Make prediction
    probabilities = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    result = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    return result, confidence


# -------------------------
# FLASK APPLICATION
# -------------------------

app = Flask(__name__)


# -------------------------
# HOME PAGE
# -------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -------------------------
# PREDICTION ROUTE
# -------------------------

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    image_path = "static/uploaded_image.jpg"

    file.save(image_path)

    result, confidence = predict_image(image_path)

    return render_template(
        "result.html",
        result=result,
        is_tumor=result != "notumor",
        confidence=confidence,
        image="uploaded_image.jpg"
    )


# -------------------------
# RUN FLASK
# -------------------------

if __name__ == "__main__":

    app.run(debug=True)