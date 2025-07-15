import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import os

# src/train.py (modified to use utils.py)
import joblib
from sklearn.linear_model import LogisticRegression
import os
# Import functions from your new utils.py
from src.utils import load_config, load_data # type: ignore
def train_model(config_path="config/config.json", model_output_path="model_train.pkl"):
    """
    Loads the digits dataset, reads hyperparameters from config.json,
    trains a LogisticRegression model, and saves it.
    """
    # Use the utility function to load config
    config = load_config(config_path)
    # Use the utility function to load data
    X, y = load_data()
    # Extract hyperparameters
    C = config.get("C", 1.0)
    solver = config.get("solver", "lbfgs")
    max_iter = config.get("max_iter", 1000)
    print(f"Training with C={C}, solver={solver}, max_iter={max_iter}")
    # Train model
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
    model.fit(X, y)
    # Save the model
    joblib.dump(model, model_output_path)
    print(f"Model trained and saved to {model_output_path}")
if __name__ == "__main__":
    # This part remains the same for creating config if not exists
    os.makedirs('config', exist_ok=True)
    if not os.path.exists("config/config.json"):
        with open("config/config.json", "w") as f:
            json.dump({"C": 1.0, "solver": "lbfgs", "max_iter": 100}, f)
    train_model()