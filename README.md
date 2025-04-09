# 🩻 Chest X-ray Tube & Catheter Positioning Error Detection

This Flask web app allows users to upload chest X-ray images and detect medical tube and catheter mispositioning using a pre-trained deep learning model in TensorFlow Lite format.

## 🧠 Features

- Upload a patient's chest X-ray image
- Select a model from a list of available `.tflite` models
- Perform inference using the selected model
- View a diagnosis table with classification probabilities
- Download diagnosis results as a CSV file

## 📂 Folder Structure


## 🚀 How to Run Locally

### 1. Clone this Repository
```bash
git clone https://github.com/elaanba/xray-diagnosis-app-1.git
cd xray-diagnosis-app


Set Up a Virtual Environment

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

Install Dependencies


pip install -r requirements.txt


Run the Flask App

python app.py





This app uses classification models that detect positioning errors for:

ETT: Endotracheal Tube

NGT: Nasogastric Tube

CVC: Central Venous Catheter

Each class includes "normal", "borderline", and "abnormal" positioning.
