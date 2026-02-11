import cv2 # pyre-ignore-all-errors
import numpy as np # pyre-ignore-all-errors
import hashlib
from skimage.feature import graycomatrix, graycoprops # pyre-ignore-all-errors

def extract_glcm_features(image_path):
    """
    Extracts GLCM texture features from a fingerprint image.
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read image")

    # Resize for consistency
    image = cv2.resize(image, (256, 256))

    # Calculate GLCM
    # Distances = 1, 3, 5 pixels
    # Angles = 0, 45, 90, 135 degrees
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(image, 
                        distances=distances, 
                        angles=angles, 
                        levels=256, 
                        symmetric=True, 
                        normed=True)

    # Extract properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    features = []
    for prop in properties:
        stats = graycoprops(glcm, prop)
        # We can take the mean across all distances and angles to simplify the feature vector
        features.append(np.mean(stats))
        # Or add individual values (3 distances * 4 angles = 12 values per property)
        # features.extend(stats.flatten())

    return np.array(features)

def analyze_fingerprint_texture(image_path):
    """
    Analyzes the physical characteristics of the fingerprint (Ridge Density & Signature).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read image")
    
    # Pre-process: Thresholding to isolate ridges
    image = cv2.resize(image, (512, 512))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 1. Ridge Density (Frequency of transitions)
    # We sample a central area to avoid background noise
    h, w = thresh.shape
    sample = thresh[h//4:3*h//4, w//4:3*w//4]
    
    # Count black-to-white transitions in the sample
    horizontal_transitions = np.sum(np.diff(sample.astype(int), axis=1) != 0) / (sample.shape[0] * sample.shape[1])
    vertical_transitions = np.sum(np.diff(sample.astype(int), axis=0) != 0) / (sample.shape[0] * sample.shape[1])
    
    ridge_density = (horizontal_transitions + vertical_transitions) / 2

    # 2. Pattern Complexity (Harris Corners as Minutiae Proxy)
    dst = cv2.cornerHarris(sample, 2, 3, 0.04)
    minutiae_density = np.sum(dst > 0.01 * dst.max()) / sample.size

    # 3. Stable Signature (Hashing)
    # We use a downsampled version of the thresholded image for a stable hash
    small = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_AREA)
    signature = hashlib.md5(small.tobytes()).hexdigest()

    return {
        "ridge_density": ridge_density,
        "minutiae_density": minutiae_density,
        "signature": signature
    }

if __name__ == "__main__":
    # Test with a dummy image if needed
    pass
