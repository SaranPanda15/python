import cv2 # pyre-ignore-all-errors
import numpy as np # pyre-ignore-all-errors
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

if __name__ == "__main__":
    # Test with a dummy image if needed
    pass
