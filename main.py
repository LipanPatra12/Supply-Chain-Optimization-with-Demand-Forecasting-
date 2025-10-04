from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
import requests
import os

# === Step 1: Download model.joblib from external link ===
MODEL_URL = "https://drive.google.com/file/d/1RnRWCbac67HKLyZenrCVLA4u1Vy80XUJ/view?usp=drive_link"  # <-- Replace with your model link

MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully.")

# === Step 2: Load encoder, scaler, and model ===
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    categorical_value = [form_data["Type"]]  # "Type" is your categorical column name

    numeric_values = []
    for key, value in form_data.items():
        if key != "Type":
            numeric_values.append(float(value))

    encoded_cat = encoder.transform([categorical_value]).toarray()
    combined_data = np.concatenate([numeric_values, encoded_cat[0]])
    scaled_data = scaler.transform([combined_data])

    prediction = model.predict(scaled_data)

    return f"Predicted value: {prediction[0]}"

if __name__ == "__main__":
    app.run(debug=True, port=5005)




