from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
model = load_model("model/plant_disease_model.h5")

print("Model loaded successfully")

# -----------------------------
# Dataset path (to get class names)
# -----------------------------
train_dir = os.path.join(
    "Dataset",
    "New Plant Diseases Dataset(Augmented)",
    "New Plant Diseases Dataset(Augmented)",
    "train"
)

# Get all disease classes
class_names = [
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
]

class_names.sort()

print("Total classes:", len(class_names))

# -----------------------------
# Upload folder
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    # Format class name nicely
    predicted_class = predicted_class.replace("___", " : ")

    return predicted_class,confidence


# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET","POST"])
def index():

    prediction = None
    img_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file.filename != "":

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            prediction = predict_disease(filepath)

            img_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        img_path=img_path
    )


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)