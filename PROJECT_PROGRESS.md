# Project Progress & Execution Log
**Project Title:** Balancing Speed and Security: A Multi-Stage Intrusion Detection System for Industrial IoT Networks  
**Institution:** Shri Vishnu Engineering College for Women (Autonomous)  
**Batch Identifier:** Batch-A10  

---

## Phase 1: Environment Setup & Dataset Verification (COMPLETED)
* **Status:** Finished
* **Actions Taken:**
  * Created the team GitHub repository (`IIoT-Major-Project`) and structured directories (`data/`, `notebooks/`, `src/`).
  * Configured local Python virtual environment (`venv`) and installed core dependencies (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `torch`, `psutil`, `joblib`, `matplotlib`, `seaborn`).
  * Implemented chunked CSV data loading (`explore_data.py`) to safely process the 3.3 GB CIC-IIoT-2025 dataset without crashing system memory.
  * **Dataset Profile Verified:**
    * Total Rows: 685,671 instances
    * Total Features: 94 columns
    * Identified Hierarchical Target Labels: `label_full`, `label1`, `label2`, `label3`, `label4`

---

## Phase 2: Data Preprocessing & Leakage-Free Splitting (COMPLETED)
* **Status:** Finished
* **Actions Taken:**
  * Cleaned dataset metadata by dropping uninformative and leak-prone identifier columns (`timestamp`, `timestamp_start`, `timestamp_end`, `device_mac`).
  * Cleaned and replaced infinite values (`inf`, `-inf`) to prevent mathematical overflow.
  * Dropped 252,852 redundant duplicate rows to eliminate data contamination and training bias.
  * Filtered out text/string features to isolate 71 pure numeric columns for robust modeling.
  * Handled missing values via median imputation across the feature space.
  * Enforced a strict, leakage-free **70/30 stratified train/test split** maintaining class distribution proportionality (~58% benign, ~42% attack).
  * Applied feature normalization using `StandardScaler` strictly fitted on the training split only, preventing data leakage into the test set.
  * **Preprocessing Outputs & Dimensions:**
    * Training Partition Shape: 302,973 rows, 71 features (`train_processed.parquet`)
    * Testing Partition Shape: 129,846 rows, 71 features (`test_processed.parquet`)
    * Fitted Scaler Artifact: Saved locally as `scaler.pkl` in the `data/` directory

---

## Phase 3: Stage 1 Edge Lightweight ML Model Selection & Training (COMPLETED)
* **Status:** Finished
* **Actions Taken:**
  * Benchmarked multiple lightweight classifiers suitable for resource-constrained IIoT edge nodes (`train_stage1_multimodel.py`).
  * Evaluated candidates on the held-out test partition (129,846 samples, 71 pure numeric features).
  * **Model Benchmarking Results:**
    * *Logistic Regression:* Accuracy: 92.37% | F1-Score: 92.42%
    * *Random Forest (Light):* Accuracy: 98.79% | F1-Score: 98.80%
    * ***Decision Tree (Winner):* Accuracy: 98.81% | Precision: 98.84% | Recall: 98.81% | F1-Score: 98.82%**
  * **Model Artifact:** Serialized and saved the winning Decision Tree classifier locally as `models/stage1_edge_model.pkl`.