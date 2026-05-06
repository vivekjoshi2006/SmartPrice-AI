import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, 
            template_folder='../templates', 
            static_folder='../static')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_path(filename):
    return os.path.join(BASE_DIR, 'models', filename)

try:
    model = joblib.load(get_model_path("house_model.pkl"))
    scaler = joblib.load(get_model_path("scaler.pkl"))
    model_cols = joblib.load(get_model_path("model_columns.pkl"))
except Exception as e:
    print(f"Model Load Error: {e}")
    model_cols = [] 

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:           
            input_values = [float(request.form.get(col, 0)) for col in model_cols]
            feature_df = pd.DataFrame([input_values], columns=model_cols)
            scaled_features = scaler.transform(feature_df)
            raw_prediction = model.predict(scaled_features)[0]
            prediction = f"${raw_prediction:,.2f}"
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, cols=model_cols)
