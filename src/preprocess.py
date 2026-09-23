import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Paths
DATA_PATH = os.path.join("data", "combined_dataset.csv")  
PROCESSED_DIR = "data"

def preprocess_pipeline():
    if not os.path.exists(DATA_PATH):
        print(f"[!] Dataset not found at {DATA_PATH}")
        return

    print("[*] Step 1: Loading dataset for preprocessing...")
    df = pd.read_csv(DATA_PATH)
    print(f"[+] Loaded dataset with shape: {df.shape}")

    # Drop uninformative or leak-prone identifier columns if present
    drop_cols = [col for col in ['timestamp', 'timestamp_start', 'timestamp_end', 'device_mac'] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[-] Dropped metadata/identifier columns: {drop_cols}")

    # Set up primary target column
    target_col = 'label1' 
    if target_col not in df.columns:
        target_col = df.columns[3]  # fallback

    print(f"[*] Step 2: Cleaning infinite values and duplicate rows...")
    # Replace infinite values with NaN so they are handled cleanly during imputation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[-] Dropped {initial_rows - len(df):,} duplicate rows to prevent training bias.")

    print(f"[*] Step 3: Handling missing values and filtering pure numeric features...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target columns and string columns from features
    target_candidates = ['label_full', 'label1', 'label2', 'label3', 'label4']
    feature_cols = [col for col in numeric_cols if col not in target_candidates]
    
    # Impute missing values on numeric feature columns only using median
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    # Separate X (features) and y (target)
    X = df[feature_cols]
    y = df[target_col]

    print(f"[*] Step 4: Executing strict stratified train/test split (70/30) to prevent data leakage...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"[+] Training set shape: {X_train.shape}")
    print(f"[+] Testing set shape: {X_test.shape}")

    print(f"[*] Step 5: Applying Feature Normalization (StandardScaler fitted on Train only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform test using train stats to avoid leakage

    # Convert back to DataFrames for convenience
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Save processed splits and scaler locally inside 'data/'
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save using parquet format for lightning-fast loading and small footprint
    X_train_df.assign(label=y_train.values).to_parquet(os.path.join(PROCESSED_DIR, "train_processed.parquet"))
    X_test_df.assign(label=y_test.values).to_parquet(os.path.join(PROCESSED_DIR, "test_processed.parquet"))
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))

    print(f"[+] Preprocessing complete! Saved processed partitions and scaler to '{PROCESSED_DIR}/'")

if __name__ == "__main__":
    preprocess_pipeline()