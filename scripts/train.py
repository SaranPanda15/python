import os
import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Add project root to sys.path to allow imports from "app"
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from app.ml.processor import extract_glcm_features

def train_models(data_dir, labels_csv, model_dir="models"):
    """
    Trains models based on real images and labels.
    
    Args:
        data_dir: Path to directory containing fingerprint images.
        labels_csv: Path to CSV file with columns: filename, blood_group, sex, age, diabetic_status
        model_dir: Directory to save trained models.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Loading labels from {labels_csv}...")
    df = pd.read_csv(labels_csv)
    
    required_cols = ["filename", "blood_group", "sex", "age", "diabetic_status"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    features_list = []
    labels = {col: [] for col in required_cols[1:]}
    
    print("Extracting features from images...")
    valid_indices = []
    
    for idx, row in df.iterrows():
        img_path = os.path.join(data_dir, row["filename"])
        if not os.path.exists(img_path):
            print(f"Warning: Image {row['filename']} not found in {data_dir}. Skipping.")
            continue
            
        try:
            features = extract_glcm_features(img_path)
            features_list.append(features)
            for col in labels:
                labels[col].append(row[col])
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    if not features_list:
        print("No valid data found to train on.")
        return

    import numpy as np
    X = np.array(features_list)
    
    for name, y in labels.items():
        print(f"Training model for {name}...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        path = os.path.join(model_dir, f"{name}_model.joblib")
        joblib.dump(model, path)
        print(f"Saved model to {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Fingerprint Analysis Models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    
    args = parser.parse_args()
    
    train_models(args.data_dir, args.labels, args.model_dir)
