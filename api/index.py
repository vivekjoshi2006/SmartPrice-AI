import os
import sys
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

app = Flask(__name__, 
            template_folder=os.path.join(root_dir, 'templates'), 
            static_folder=os.path.join(root_dir, 'static'))

def load_model_files():
    try:
        models_dir = os.path.join(root_dir, 'models')
        
        m = joblib.load(os.path.join(models_dir, "house_model.pkl"))
        s = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        c = joblib.load(os.path.join(models_dir, "model_columns.pkl"))
        return m, s, c
    except Exception as e:
        print(f"FAILED TO LOAD MODELS: {str(e)}")
        return None, None, []

model, scaler, model_cols = load_model_files()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            if model is None:
                return "Error: AI Model could not be loaded on server. Check logs.", 500

            input_values = [float(request.form.get(col, 0)) for col in model_cols]
            df = pd.DataFrame([input_values], columns=model_cols)

            scaled = scaler.transform(df)
            res = model.predict(scaled)[0]
            prediction = f"${res:,.2f}"
        except Exception as e:
            prediction = f"Calculation Error: {str(e)}"

    return render_template("index.html", prediction=prediction, cols=model_cols)

app = app
