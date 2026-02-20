import os
import sys
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path to allow imports from "app"
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.append(str(ROOT_PATH))

from app.ml.processor import extract_glcm_features

def train_models(data_dir, labels_csv, model_dir="models"):
    """
    Trains models based on real images and labels.
    """
    data_path = Path(data_dir)
    labels_path = Path(labels_csv)
    model_path = Path(model_dir)

    if not labels_path.exists():
        print(f"\nError: Labels file not found at {labels_csv}")
        print("To generate a dummy labels file for testing, run:")
        print("python scripts/create_labels.py\n")
        return

    if not data_path.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels from {labels_csv}...")
    df = pd.read_csv(labels_csv)
    
    required_cols = ["filename", "blood_group", "sex", "age", "diabetic_status"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: CSV must contain column: {col}")
            return

    features_list = []
    labels = {col: [] for col in required_cols[1:]}
    
    print("Extracting features from images...")
    
    for idx, row in df.iterrows():
        img_path = data_path / row["filename"]
        if not img_path.exists():
            print(f"Warning: Image {row['filename']} not found in {data_dir}. Skipping.")
            continue
            
        try:
            features = extract_glcm_features(str(img_path))
            features_list.append(features)
            for col in labels:
                labels[col].append(row[col])
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    if not features_list:
        print("No valid data found to train on. Check your images and labels.csv.")
        return

    X = np.array(features_list)
    
    for name, y in labels.items():
        print(f"Training model for {name}...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        save_path = model_path / f"{name}_model.joblib"
        joblib.dump(model, str(save_path))
        print(f"Saved model to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Fingerprint Analysis Models")
    parser.add_argument("--data_dir", type=str, default="uploads", help="Directory containing images")
    parser.add_argument("--labels", type=str, default="labels.csv", help="Path to labels CSV")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    
    args = parser.parse_args()
    
    train_models(args.data_dir, args.labels, args.model_dir)
