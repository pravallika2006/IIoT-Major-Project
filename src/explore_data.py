import pandas as pd
import os

# Define path to dataset (assumes you place your CSV inside the 'data' folder)
DATA_PATH = os.path.join("data", "combined_dataset.csv")

def verify_dataset_schema():
    if not os.path.exists(DATA_PATH):
        print(f"[!] Dataset not found at '{DATA_PATH}'.")
        print("Please place your 'combined_dataset.csv' inside the 'data/' folder.")
        return

    print("[*] Loading dataset chunks to inspect schema safely...")
    chunk_size = 100000
    total_rows = 0
    sample_df = None

    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):
        if sample_df is None:
            sample_df = chunk.head(5)
        total_rows += len(chunk)

    print(f"\n[+] Total Rows Verified: {total_rows:,}")
    print(f"[+] Total Features (Columns): {sample_df.shape[1]}")
    print(f"\n[+] Column Names Preview:")
    print(list(sample_df.columns))

    # Check hierarchical target labels if present
    target_cols = [col for col in sample_df.columns if 'label' in col.lower()]
    print(f"\n[+] Potential Target Columns Identified: {target_cols}")

if __name__ == "__main__":
    verify_dataset_schema()