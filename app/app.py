from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import datetime
import os
import re

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
data_model = joblib.load("model/leakage_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
anomaly_model = joblib.load("model/anomaly_model.pkl")

# ---------------- RISK MAPPING ----------------
data_risk_map = {
    "Public": 0,
    "Internal": 2,
    "Confidential": 4,
    "Highly_Sensitive": 6
}

LOG_FILE = "logs/risk_logs.csv"

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- HELPER: FILE SCAN ----------------
def scan_file(file_path):
    with open(file_path, "r", errors="ignore") as f:
        content = f.read()

    content_length = len(content)

    contains_email = 1 if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", content) else 0
    contains_pan = 1 if re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", content) else 0
    contains_aadhaar = 1 if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", content) else 0

    return content_length, contains_email, contains_pan, contains_aadhaar

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    # ---------- FILE UPLOAD ----------
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)

    content_length, contains_email, contains_pan, contains_aadhaar = scan_file(file_path)

    # ---------- USER INPUT ----------
    try:
        file_type_encoded = int(request.form["file_type_encoded"])
        downloads_mb = float(request.form["downloads_mb"])
        uploads_mb = float(request.form["uploads_mb"])
        login_hour = int(request.form["login_hour"])
        access_frequency = int(request.form["access_frequency"])
    except:
        return jsonify({"error": "Invalid user input"}), 400

    if not (0 <= login_hour <= 23):
        return jsonify({"error": "Login hour must be between 0 and 23"}), 400

    # ---------- ML INPUT ----------
    file_df = pd.DataFrame([{
        "content_length": content_length,
        "contains_email": contains_email,
        "contains_pan": contains_pan,
        "contains_aadhaar": contains_aadhaar,
        "file_type_encoded": file_type_encoded
    }])

    user_behavior = [[downloads_mb, uploads_mb, login_hour, access_frequency]]

    # ---------- PREDICTIONS ----------
    data_pred = data_model.predict(file_df)
    data_label = label_encoder.inverse_transform(data_pred)[0]
    behavior_flag = anomaly_model.predict(user_behavior)[0]

    data_risk = data_risk_map[data_label]
    behavior_risk = 5 if behavior_flag == -1 else 0
    total_risk = data_risk + behavior_risk

    decision = (
        "HIGH RISK" if total_risk >= 8 else
        "MEDIUM RISK" if total_risk >= 4 else
        "LOW RISK"
    )

    # ---------- LOG ----------
    with open(LOG_FILE, "a") as f:
        f.write(
            f"{datetime.now()},{content_length},{contains_email},{contains_pan},{contains_aadhaar},"
            f"{file_type_encoded},{downloads_mb},{uploads_mb},{login_hour},{access_frequency},"
            f"{data_label},{behavior_flag},{total_risk},{decision}\n"
        )

    return jsonify({
        "data_sensitivity": data_label,
        "risk_score": total_risk,
        "decision": decision,
        "detected": {
            "email": contains_email,
            "pan": contains_pan,
            "aadhaar": contains_aadhaar
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
