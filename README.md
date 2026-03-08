# 🌿 Plant Disease Detection using Deep Learning

This project is a deep learning-based web application that detects plant diseases from leaf images.  
The model is trained using **Transfer Learning with MobileNetV2** and deployed using a **Flask web application** where users can upload a leaf image and get a predicted disease.

---

# 📌 Project Overview

Plant diseases can significantly affect crop production and agricultural productivity. Early detection helps farmers take preventive actions and improve crop yield.

This project uses a **Convolutional Neural Network (CNN)** model trained on the **PlantVillage dataset** to identify plant diseases from leaf images.

Users can upload a leaf image through the web interface and the system predicts the disease.

---

# 🚀 Features

- Deep learning model trained using **TensorFlow/Keras**
- Transfer learning with **MobileNetV2**
- Image classification of plant diseases
- Flask-based web application
- Upload leaf image and get prediction instantly
- Displays disease prediction with confidence score

---

# 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Matplotlib
- HTML
- CSS

---

# 📂 Project Structure


Plant-Disease-Detection
│
├── app.py
├── requirements.txt
├── README.md
│
├── model
│ └── plant_disease_model.h5
│
├── templates
│ └── index.html
│
├── static
│ ├── css
│ │ └── style.css
│ └── uploads
│
├── notebook
│ └── Plant_Disease_Detection.ipynb
│
└── Dataset (Not included in repo)


---

# 📊 Model Details

- Model Architecture: **MobileNetV2**
- Image Size: **224 × 224**
- Training Accuracy: **~97%**
- Validation Accuracy: **~96%**
- Number of Classes: **38 plant disease classes**

---

# 🌱 Dataset

The model was trained using the **PlantVillage dataset** available on Kaggle.

Dataset link:

https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

The dataset contains:

- 70,000+ leaf images
- 38 plant disease categories
- Images of healthy and diseased leaves

Because the dataset is large, it is **not included in this repository**.

To download the dataset:


Go to the Kaggle dataset link above

Download the dataset

Extract it into the project folder as:

Dataset/
New Plant Diseases Dataset(Augmented)/
New Plant Diseases Dataset(Augmented)/
train/
valid/


---

# ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/Antonio-rohit/Plant-Disease-Detection.git

Navigate to the project folder:

cd Plant-Disease-Detection

Install dependencies:

pip install -r requirements.txt
▶ Run the Application

Run the Flask application:

python app.py

Open your browser and go to:

http://127.0.0.1:5000

Upload a leaf image to get the predicted plant disease.

🖼 Example Prediction

Example output:

Prediction: Squash : Powdery Mildew
Confidence: 96%
🔮 Future Improvements

Deploy the application online (Render / Heroku)

Add disease treatment recommendations

Improve UI design

Expand dataset for more crops

Convert model to a mobile app

👨‍💻 Author

Rohit Bharwade

⭐ If you like this project

Give the repository a star ⭐ on GitHub.


---

After creating this file:

1️⃣ Save it as **README.md** in your project root  
2️⃣ Upload it:

```bash
git add README.md
git commit -m "Added project README"
git push


Document:-
https://drive.google.com/drive/folders/1PgjFQylsDFVvc2oV3DHcOjuW9hwtkWbl