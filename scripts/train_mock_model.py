import joblib # pyre-ignore-all-errors
import numpy as np # pyre-ignore-all-errors
import os # pyre-ignore-all-errors
from sklearn.ensemble import RandomForestClassifier # pyre-ignore-all-errors

def train_mock_models():
    """
    Trains and saves mock RandomForest models for demonstration.
    """
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Features: 5 GLCM features (contrast, dissimilarity, homogeneity, energy, correlation)
    num_samples = 100
    X = np.random.rand(num_samples, 5)

    # Mock targets
    targets = {
        "blood_group": np.random.choice(["A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"], num_samples),
        "sex": np.random.choice(["Male", "Female"], num_samples),
        "age": np.random.choice(["Young", "Adult", "Senior"], num_samples),
        "diabetic_status": np.random.choice(["Diabetic", "Non-Diabetic"], num_samples)
    }

    for name, y in targets.items():
        model = RandomForestClassifier(n_estimators=10)
        model.fit(X, y)
        path = os.path.join(model_dir, f"{name}_model.joblib")
        joblib.dump(model, path)
        print(f"Saved mock model for {name} to {path}")

if __name__ == "__main__":
    train_mock_models()
