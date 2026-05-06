import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, 
            template_folder='../templates', 
            static_folder='../static')

ROOT_PATH = os.getcwd()

def load_pkl(filename):
    paths_to_check = [
        os.path.join(ROOT_PATH, 'models', filename),
        os.path.join(ROOT_PATH, filename),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', filename))
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError(f"Could not find {filename} in any known directory")

try:
    model = load_pkl("house_model.pkl")
    scaler = load_pkl("scaler.pkl")
    model_cols = load_pkl("model_columns.pkl")
except Exception as e:
    print(f"CRITICAL MODEL LOAD ERROR: {e}")
    model_cols = [] 

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            if not model_cols:
                return "Models not loaded correctly on server.", 500
                
            input_values = [float(request.form.get(col, 0)) for col in model_cols]
            feature_df = pd.DataFrame([input_values], columns=model_cols)
            scaled_features = scaler.transform(feature_df)
            raw_prediction = model.predict(scaled_features)[0]
            prediction = f"${raw_prediction:,.2f}"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, cols=model_cols)

app = app
