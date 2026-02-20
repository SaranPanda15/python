# Fingerprint-Based Physiological Analysis System

## 🚀 Project Overview
This project is a sophisticated **Biometric Analysis System** built with **FastAPI** and **Machine Learning**. It processes fingerprint images to predict physiological attributes such as **Blood Group, Sex, Age, and Diabetic Status**. The system utilizes advanced image processing techniques like **GLCM (Gray-Level Co-occurrence Matrix)** and structural ridge analysis to extract meaningful features from biometric data.

---

## 🏗️ Project Structure
```text
.
├── app/                    # Main application directory
│   ├── main.py             # FastAPI entry point & API endpoints
│   ├── ml/                 # Machine Learning & Image Processing logic
│   │   ├── processor.py    # Feature extraction (GLCM, Ridge Density)
│   │   └── model_manager.py# Model loading, inference & fallback engine
│   ├── static/             # Static assets (CSS, JS, Images)
│   └── templates/          # Jinja2 HTML templates
├── models/                 # Pre-trained ML models (.joblib)
├── scripts/                # Utility scripts for training & testing
├── uploads/                # Temporary storage for analyzed images
├── labels.csv              # Ground truth data for training/verification
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Technical Stack
- **Backend**: FastAPI (Python)
- **Image Processing**: OpenCV, Scikit-Image, NumPy
- **Machine Learning**: Scikit-Learn
- **Frontend**: Jinja2 Templates, HTML5, Vanilla CSS
- **Data Handling**: Pandas, Joblib
- **Security**: SHA-256/MD5 Hashing for Biometric Signatures

---

## 🧠 Machine Learning & Technical Analysis

### 1. Feature Extraction Methodology
The core of the system lies in its ability to convert a fingerprint image into a numerical "Biometric Signature." We utilize three primary methods:
- **GLCM (Gray-Level Co-occurrence Matrix)**: Extracts texture features by calculating how often pairs of pixels with specific values and in a specified spatial relationship occur in an image.
  - *Features*: Contrast, Dissimilarity, Homogeneity, Energy, and Correlation.
- **Ridge Density Analysis**: Calculates the frequency of black-to-white transitions. Research suggests that ridge density correlates with gender and age.
- **Minutiae Proxy (Harris Corners)**: Detects corner-like structures (ridge endings and bifurcations) to estimate pattern complexity.

### 2. Machine Learning Models
The system is designed to use **Ensemble Learning** (specifically Random Forest or Gradient Boosting) to classify the extracted features. The models are:
- **`blood_group_model.joblib`**: Predicts A/B/O/AB groups based on texture patterns.
- **`sex_model.joblib`**: Uses ridge density and GLCM features (Females typically have higher ridge density).
- **`age_model.joblib`**: Analyzes ridge thickness and texture degradation.
- **`diabetic_status_model.joblib`**: Experimental model correlating minutiae density with physiological health.

### 3. DeepRidge™ Fallback Engine
In cases where pre-trained models are unavailable, the system utilizes a **Heuristic Engine** that applies rule-based logic derived from forensic literature:
- **Sex Prediction**: Higher ridge density (>0.08) defaults to Female.
- **Age Prediction**: Categorized based on ridge transition frequency (Young/Adult/Senior).
- **Diabetic Status**: Correlated with minutiae density thresholds.

---

## 🔍 Validation & Security
The system includes a **Biometric Validator** in `processor.py` that ensures:
- The image is not blank (Variance check).
- The image has a biometric-like structure (Blur stability test).
- The ridge frequency falls within human physiological bounds.

---

## 🛠️ Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   python app/main.py
   ```

3. **Inference**:
   - Access `http://localhost:8000` via browser.
   - Upload a `.png` or `.jpg` fingerprint image.
   - View the real-time AI analysis.

---

## 🎓 Student Resources
For students using this project:
- Refer to the **SRS.md** for detailed software requirements.
- Explore `app/ml/processor.py` to understand digital image processing.
- Study `app/ml/model_manager.py` to see how Python handles multi-model inference.
