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

## Phase 2: Data Preprocessing & Leakage-Free Splitting (UPCOMING)
* **Status:** Pending
* **Planned Tasks:**
  * Handle missing values and outliers (IQR capping).
  * Enforce strict stratified train/test splitting before feature selection to avoid data leakage.
  * Apply feature scaling/normalization.