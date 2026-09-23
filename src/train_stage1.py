import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"

def train_stage1_model():
    print("[*] Step 1: Loading preprocessed training and testing partitions...")
    train_path = os.path.join(DATA_DIR, "train_processed.parquet")
    test_path = os.path.join(DATA_DIR, "test_processed.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[!] Processed parquet files not found! Please run 'preprocess.py' first.")
        return

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Separate features and target ('label')
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    print(f"[+] Training data loaded. Features: {X_train.shape[1]}, Samples: {X_train.shape[0]:,}")
    print(f"[+] Testing data loaded. Samples: {X_test.shape[0]:,}")

    print("[*] Step 2: Training Stage 1 Edge Lightweight Classifier (Logistic Regression)...")
    # Using a fast, memory-efficient linear model optimized for edge devices
    stage1_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    stage1_model.fit(X_train, y_train)

    print("[*] Step 3: Evaluating model performance on test partition...")
    y_pred = stage1_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print(f"\n[+] --- Stage 1 Evaluation Metrics ---")
    print(f"    * Accuracy:  {acc * 100:.2f}%")
    print(f"    * Precision: {precision * 100:.2f}%")
    print(f"    * Recall:    {recall * 100:.2f}%")
    print(f"    * F1-Score:  {f1 * 100:.2f}%")
    
    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred))

    print("[*] Step 4: Saving Stage 1 model artifact...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "stage1_edge_model.pkl")
    joblib.dump(stage1_model, model_path)
    print(f"[+] Stage 1 model successfully saved to '{model_path}'")

if __name__ == "__main__":
    train_stage1_model()