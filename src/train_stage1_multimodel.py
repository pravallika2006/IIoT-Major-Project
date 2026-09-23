import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"

def evaluate_edge_models():
    print("[*] Loading preprocessed training and testing partitions...")
    train_path = os.path.join(DATA_DIR, "train_processed.parquet")
    test_path = os.path.join(DATA_DIR, "test_processed.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest (Light)": RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    }

    results = []
    trained_models = {}

    print("\n[*] Training and benchmarking candidate models for Stage 1 Edge...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in candidates.items():
        print(f"    -> Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    results_df = pd.DataFrame(results)
    print("\n[+] --- Stage 1 Model Benchmarking Results ---")
    print(results_df.to_string(index=False))

    # Select the best performing model based on F1-Score
    best_row = results_df.loc[results_df['F1-Score'].idxmax()]
    best_model_name = best_row['Model']
    best_model = trained_models[best_model_name]
    
    print(f"\n[+] Best Performing Edge Model: {best_model_name} with F1-Score: {best_row['F1-Score']*100:.2f}%")

    # Save the winner as the official Stage 1 deployment artifact
    official_model_path = os.path.join(MODEL_DIR, "stage1_edge_model.pkl")
    joblib.dump(best_model, official_model_path)
    print(f"[+] Official Stage 1 model successfully saved to '{official_model_path}'")

if __name__ == "__main__":
    evaluate_edge_models()