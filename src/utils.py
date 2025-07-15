import json
from sklearn.datasets import load_digits

def load_config(config_path="config/config.json"):
    """
    Loads configuration from a JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {config_path}")
    return config

def load_data():
    """
    Loads the digits dataset from sklearn.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    print("Digits dataset loaded.")
    return X, y