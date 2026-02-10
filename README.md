# Fingerprint Analysis API

A FastAPI-based system to analyze fingerprint images and predict user attributes (Blood Group, Sex, Age, Diabetic Status) using GLCM texture features and Random Forest models.

## Features
- **Biometric Attribute Prediction**: Predicts age, sex, blood group, and diabetic status.
- **FastAPI Backend**: High-performance API with automatic documentation.
- **Biometric Feature Extraction**: Uses Gray-Level Co-occurrence Matrix (GLCM) for texture analysis.
- **Modern UI**: Clean, responsive interface for uploading and analyzing fingerprints.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**:
   
   If you have a dataset (images and a `labels.csv` file), run:
   ```bash
   python scripts/train.py --data_dir path/to/images --labels path/to/labels.csv
   ```
   
   The `labels.csv` should have the following columns: 
   `filename, blood_group, sex, age, diabetic_status`

   For a quick demonstration with random data, you can still use:
   ```bash
   python scripts/train_mock_model.py
   ```

3. **Run the Application**:
   ```bash
   python app/main.py
   ```
   Access the UI at `http://localhost:8000`

## Project Structure
- `app/main.py`: Main API entry point.
- `app/ml/processor.py`: GLCM feature extraction logic.
- `app/ml/model_manager.py`: Model loading and inference.
- `app/templates/index.html`: Web interface.
- `models/`: Directory where trained models are stored.
