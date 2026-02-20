import os
import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from app.ml.model_manager import manager

def test_lookup():
    # We know F1.png should be in uploads and potentially in labels.csv
    # But let's check one from the csv
    import pandas as pd
    df = pd.read_csv(root_path / "labels.csv")
    if df.empty:
        print("labels.csv is empty")
        return
    
    test_file = df.iloc[0]["filename"]
    print(f"Testing lookup for: {test_file}")
    
    # Simulate a full path (even though lookup uses basename)
    full_path = root_path / "uploads" / test_file
    
    if not full_path.exists():
        print(f"File {full_path} not found for testing.")
        return

    result = manager.predict(str(full_path))
    print("Result:", result)
    
    if result.get("source") == "Training Data (Ground Truth)":
        print("SUCCESS: Lookup worked!")
    else:
        print("FAILURE: Lookup did not work.")

if __name__ == "__main__":
    test_lookup()
