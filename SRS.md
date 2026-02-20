# Software Requirements Specification (SRS)
## Project: Fingerprint-Based Physiological Analysis System

### 1. Introduction
#### 1.1 Purpose
This document specifies the requirements for a Fingerprint Analysis System that predicts physiological traits (Blood Group, Sex, Age, Diabetic Status) from digital fingerprint images. It serves as a guide for developers, testers, and academic reviewers.

#### 1.2 Scope
The system provides a web-based interface for uploading fingerprint images, performs automated feature extraction, and uses machine learning models to provide analytical predictions.

---

### 2. Overall Description
#### 2.1 System Environment
- **Platform**: Cross-platform (Windows/Linux/macOS)
- **Architecture**: Client-Server (REST API using FastAPI)
- **Input**: Fingerprint image (JPEG/PNG)
- **Output**: JSON/HTML report containing physiological predictions

#### 2.2 User Classes and Characteristics
- **Student/Researcher**: Uses the tool to study biometric correlations.
- **Developer**: Extends the system with new ML models or features.

---

### 3. Functional Requirements

#### 3.1 Image Upload & Validation
- **FR.1**: The system shall accept image files (PNG, JPG, BMP).
- **FR.2**: The system shall validate if the uploaded image contains a recognizable fingerprint pattern.
- **FR.3**: The system shall reject images with insufficient contrast or high noise levels.

#### 3.2 Feature Extraction
- **FR.4**: The system shall extract GLCM (Gray-Level Co-occurrence Matrix) features (Contrast, Homogeneity, etc.).
- **FR.5**: The system shall calculate Ridge Density and Minutiae Density.
- **FR.6**: The system shall generate a unique MD5-based biometric signature for internal tracking.

#### 3.3 Prediction & Analysis
- **FR.7**: The system shall use pre-trained Joblib models for attribute classification.
- **FR.8**: The system shall provide a heuristic fallback engine if ML models are missing.
- **FR.9**: The system shall return a confidence-based or rule-based result for Sex, Age, Blood Group, and Diabetic Status.

---

### 4. External Interface Requirements

#### 4.1 User Interface
- A clean, modern dashboard for file selection.
- Real-time display of analysis results without page refreshes.
- Error messaging for invalid file types or failed biometric analysis.

#### 4.2 API Interface
- `GET /`: Serves the frontend.
- `POST /analyze`: Accepts a multipart file and returns analysis results.

---

### 5. Technical Requirements (System Attributes)

#### 5.1 Reliability
- The system should handle corrupted images gracefully without crashing the server.

#### 5.2 Performance
- Feature extraction and inference should complete within < 2 seconds for a standard 512x512 image.

#### 5.3 Security
- No biometric data should be stored permanently unless explicitly configured (privacy by design).
- Hashed signatures are used to identify patterns without revealing raw fingerprint data.

---

### 6. Design Constraints & Tech Stack
- **Language**: Python 3.8+
- **Framework**: FastAPI
- **Libraries**:
  - OpenCV: For image pre-processing.
  - Scikit-Image: For GLCM analysis.
  - Scikit-Learn: For classification models.
- **Frontend**: HTML5, Vanilla CSS, Jinja2.

---

### 7. Appendix
#### 7.1 Glossary
- **GLCM**: Gray-Level Co-occurrence Matrix.
- **Minutiae**: Small details in a fingerprint (ridge endings/bifurcations).
- **Ridge Density**: Number of ridges in a certain area of the fingerprint.
