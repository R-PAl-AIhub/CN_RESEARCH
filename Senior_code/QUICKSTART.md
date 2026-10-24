# QUICK START GUIDE
# Federated Learning for DDoS Detection in IoT Networks

## For Immediate Testing (Without CIC-IoT2023 Dataset)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure for Synthetic Data
Edit `config.py` line 19:
```python
USE_REAL_DATASET = False  # Set to False for demo mode
```

### 3. Run Complete Experiment
```bash
python main_experiment.py
```

Expected output: Complete experiment with synthetic data (~5-10 minutes)

### 4. Generate Figures
```bash
python visualization.py
```

---

## For Paper Reproduction (With CIC-IoT2023 Dataset)

### 1. Download Dataset
- Source: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Extract all CSV files to: `./data/cic-iot-2023/`

### 2. Configure for Real Data
Edit `config.py` line 19:
```python
USE_REAL_DATASET = True  # Set to True for real data
DATASET_PATH = "./data/cic-iot-2023"  # Adjust path if needed
```

### 3. Run Complete Experiment
```bash
python main_experiment.py
```

Expected runtime: 30-60 minutes for 500,000 samples

### 4. Generate Figures
```bash
python visualization.py
```

---

## Individual Modules

### Run Only Preprocessing (Algorithm 1)
```bash
python data_preprocessing.py
```

### Run Only Federated Training (Algorithm 2)
```bash
python federated_learning.py
```

### Generate Visualizations Only
```bash
python visualization.py
```

---

## Expected Outputs

### Console Output
- Detailed progress logs
- Algorithm 1: Feature selection steps (46 → 17 features)
- Algorithm 2: Federated training rounds
- Performance tables (Tables 2-4 from paper)

### Files Generated
```
outputs/
├── results/
│   ├── experiment_results.json          # Complete results
│   ├── table2_federated_vs_centralized.csv
│   └── table3_algorithm_comparison.csv
├── figures/
│   ├── figure3_performance_comparison.png
│   ├── figure4_model_comparison.png
│   ├── figure5_confusion_matrix.png
│   └── figure6_scalability_analysis.png
└── data/
    ├── X_train.npy, X_test.npy
    ├── y_train.npy, y_test.npy
    └── feature_names.txt
```

---

## Troubleshooting

### Issue: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Issue: "No CSV files found"
- Check dataset path in config.py
- OR use synthetic mode: `USE_REAL_DATASET = False`

### Issue: Memory error
- Reduce samples: `REAL_DATASET_SAMPLES = 100000` in config.py
- Use synthetic mode for testing

---

## Verification

### Paper Results to Match:
- Features: 46 → 17 (63% reduction) ✓
- Federated SVM (5 workers) F1-Score: 0.9881 ✓
- Performance Parity: Fed. = Centralized ✓
- FPR: 0.34% ✓

Check console output and `outputs/results/` files!

---

## Contact
Author: Haris Mohamed C P
Email: harismohamedcp@gmail.com
