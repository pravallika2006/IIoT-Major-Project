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
  * Cleaned dataset metadata by dropping uninformative and leak-prone identifier columns (`timestamp`, `timestamp_start`, `timestamp_end`, `device_mac`)[cite: 3, 15].
  * Filtered out text/string features to isolate 71 pure numeric columns for robust modeling[cite: 15].
  * Handled missing values via median imputation across the feature space[cite: 15].
  * Enforced a strict, leakage-free **70/30 stratified train/test split** maintaining class distribution proportionality[cite: 15].
  * Applied feature normalization using `StandardScaler` strictly fitted on the training split only, preventing data leakage into the test set[cite: 15].
  * **Preprocessing Outputs & Dimensions:**
    * Training Partition Shape: 479,969 rows, 71 features (`train_processed.parquet`)[cite: 15]
    * Testing Partition Shape: 205,702 rows, 71 features (`test_processed.parquet`)[cite: 15]
    * Fitted Scaler Artifact: Saved locally as `scaler.pkl` in the `data/` directory[cite: 15]