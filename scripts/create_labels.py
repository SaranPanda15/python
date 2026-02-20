import os
import pandas as pd
import random
from pathlib import Path

def create_dummy_labels(data_dir="uploads", output_file="labels.csv"):
    """
    Scans data_dir for images and creates a labels.csv with randomized data.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: {data_dir} directory not found.")
        return

    images = [f.name for f in data_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    if not images:
        print(f"No images found in {data_dir}.")
        return

    data = []
    blood_groups = ["A+", "B+", "O+", "AB+", "A-", "B-", "O-", "AB-"]
    sexes = ["Male", "Female"]
    ages = ["Young", "Adult", "Senior"]
    diabetic_statuses = ["Diabetic", "Non-Diabetic"]

    for img in images:
        data.append({
            "filename": img,
            "blood_group": random.choice(blood_groups),
            "sex": random.choice(sexes),
            "age": random.choice(ages),
            "diabetic_status": random.choice(diabetic_statuses)
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(images)} entries.")

if __name__ == "__main__":
    create_dummy_labels()
