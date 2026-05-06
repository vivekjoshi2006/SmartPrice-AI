import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def train_house_model():
    """
    Trains a Random Forest Regressor to predict house prices,
    saves the model, scaler, and column list for future inference.
    """
    # Create directory for saving models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("--- Created 'models' directory ---")

    # Load dataset
    try:
        data = pd.read_csv("house_data.csv")
        print(f"--- Dataset Loaded: {len(data)} rows found ---")
    except FileNotFoundError:
        print("Error: 'house_data.csv' not found! Please check the file path.")
        return

    # Data Preprocessing
    data = data.select_dtypes(include=[np.number])
    if data.isnull().values.any():
        data = data.fillna(data.mean())
        print("--- Handled missing values using column means ---")

    # Check for target column
    if 'price' not in data.columns:
        print("Error: Column 'price' not found in the dataset!")
        return

    # Splitting Features and Target
    X = data.drop("price", axis=1)
    y = data["price"]

    # Save feature names to ensure consistency during prediction
    model_cols = list(X.columns)
    joblib.dump(model_cols, "models/model_columns.pkl")
    print(f"1. Saved features list: {model_cols}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, "models/scaler.pkl")
    print("2. Saved scaler.pkl")

    # Model Training
    print("Training the Random Forest model (this may take a few seconds)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the trained model
    joblib.dump(model, "models/house_model.pkl")
    print("3. Saved house_model.pkl")

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n--- Training Complete ---")
    print(f"Model Accuracy (R2 Score): {round(r2 * 100, 2)}%")
    print(f"Average Prediction Error: ${round(mae, 2)}")

if __name__ == "__main__":
    train_house_model()
