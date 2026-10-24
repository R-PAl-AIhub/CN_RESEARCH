# Privacy-Preserving Federated Learning for DDoS Detection in IoT Networks

## Paper Information

**Title**: Privacy-Preserving Federated Learning Framework for DDoS Attack Detection in IoT Networks Using Hybrid Feature Selection

**Author**: Haris Mohamed C P  
**Affiliation**: Department of Computer Science and Engineering, Vellore Institute of Technology, Vellore, Tamil Nadu, India  
**Email**: harismohamedcp@gmail.com

**Paper Reference**: This codebase implements Algorithms 1 and 2 from the paper, reproducing all experimental results (Tables 2-4, Figures 3-6).

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Running the Experiments](#running-the-experiments)
6. [Reproducing Paper Results](#reproducing-paper-results)
7. [Module Documentation](#module-documentation)
8. [Output Files](#output-files)
9. [Hardware Requirements](#hardware-requirements)
10. [Troubleshooting](#troubleshooting)
11. [Citation](#citation)

---

## Overview

This repository contains the complete implementation of the privacy-preserving federated learning framework for DDoS attack detection in IoT networks. The code reproduces all experimental results presented in the paper, including:

- **Algorithm 1**: Hybrid Feature Selection (Pearson ∩ LDA)
- **Algorithm 2**: Federated SVM Training
- **Table 2**: Federated vs Centralized Performance
- **Table 3**: Algorithm Comparison (5 Workers)
- **Table 4**: Confusion Matrix Analysis
- **Figures 3-6**: Performance visualizations

### Key Features

✅ **Modular Architecture**: Clean separation of concerns (config, data preprocessing, federated learning, visualization)  
✅ **Reproducibility**: Fixed random seeds (RANDOM_SEED=42) for consistent results  
✅ **Scalability**: Supports multiple worker configurations (3, 5, 10 workers)  
✅ **Comprehensive Evaluation**: ML metrics + physical metrics (training time, memory)  
✅ **Visualization**: Automated generation of publication-quality figures  
✅ **Demo Mode**: Synthetic data generation for testing without CIC-IoT2023 dataset  

---

## Repository Structure

```
federated-ddos-detection/
│
├── config.py                   # Configuration module (all parameters)
├── data_preprocessing.py       # Algorithm 1: Data loading + feature selection
├── federated_learning.py       # Algorithm 2: Federated SVM + comparative models
├── main_experiment.py          # Main experiment runner
├── visualization.py            # Figure generation module
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── data/                       # Dataset directory (user-provided)
│   └── cic-iot-2023/          # Place CIC-IoT2023 CSV files here
│
└── outputs/                    # Experiment outputs (auto-generated)
    ├── results/               # JSON and CSV results
    │   ├── experiment_results.json
    │   ├── table2_federated_vs_centralized.csv
    │   └── table3_algorithm_comparison.csv
    ├── figures/               # Generated figures (PNG)
    │   ├── figure3_performance_comparison.png
    │   ├── figure4_model_comparison.png
    │   ├── figure5_confusion_matrix.png
    │   └── figure6_scalability_analysis.png
    ├── data/                  # Preprocessed data (NPY files)
    │   ├── X_train.npy
    │   ├── X_test.npy
    │   ├── y_train.npy
    │   ├── y_test.npy
    │   └── feature_names.txt
    └── models/                # Saved models (optional)
```

---

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 8GB (16GB recommended for real dataset)
- **Storage**: Minimum 20GB free space (for CIC-IoT2023 dataset)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd federated-ddos-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n federated-ddos python=3.10
conda activate federated-ddos
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt Contents

```
numpy>=1.24.3
pandas>=2.0.3
matplotlib>=3.7.2
seaborn>=0.12.2
scikit-learn>=1.3.0
scipy>=1.11.0
```

### Step 4: Verify Installation

```bash
python config.py
```

Expected output:
```
================================================================================
CONFIGURATION MODULE
================================================================================
Random Seed: 42
Use Real Dataset: True
...
✓ Configuration loaded successfully
```

---

## Dataset Preparation

### Option 1: Using Real CIC-IoT2023 Dataset (Recommended for Paper Reproduction)

1. **Download CIC-IoT2023 Dataset**
   - Source: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
   - Size: ~13GB compressed, ~46 million records

2. **Extract Dataset**
   ```bash
   mkdir -p data/cic-iot-2023
   # Extract all CSV files to data/cic-iot-2023/
   ```

3. **Update Configuration**
   
   Edit `config.py`:
   ```python
   USE_REAL_DATASET = True
   DATASET_PATH = "./data/cic-iot-2023"  # Adjust to your path
   ```

4. **Verify Dataset**
   ```bash
   ls data/cic-iot-2023/*.csv
   ```
   
   Should show multiple CSV files from the CIC-IoT2023 collection.

### Option 2: Using Synthetic Data (Demo Mode)

If you don't have access to CIC-IoT2023, the code can generate synthetic data for testing:

1. Edit `config.py`:
   ```python
   USE_REAL_DATASET = False  # Enable synthetic mode
   ```

2. Run experiments (synthetic data will be auto-generated)

**Note**: Synthetic data is for demonstration only. Results will differ from paper.

---

## Running the Experiments

### Complete Experiment Pipeline (Reproduces All Paper Results)

```bash
python main_experiment.py
```

This single command executes:
1. ✅ Data loading and preprocessing (Algorithm 1)
2. ✅ Centralized SVM baseline training
3. ✅ Federated SVM training (3, 5, 10 workers) - Algorithm 2
4. ✅ Comparative models (RF, ANN, IF)
5. ✅ Results evaluation and saving

**Expected Runtime**: 
- Synthetic data: ~5-10 minutes
- Real data (500K samples): ~30-60 minutes (depends on CPU)

**Output**: 
- Console: Detailed progress and results tables
- Files: `outputs/results/experiment_results.json` + CSV tables

### Step-by-Step Execution (Individual Modules)

#### Step 1: Data Preprocessing Only

```bash
python data_preprocessing.py
```

This executes **Algorithm 1** (Hybrid Feature Selection):
- Loads CIC-IoT2023 or generates synthetic data
- Applies variance filtering (Step 1)
- Pearson correlation selection (Step 2)
- LDA feature selection (Step 3)
- Feature intersection (Step 4)
- Cross-validation check (Step 5)
- Train-test split (80-20 stratified)

**Output**:
- `outputs/data/X_train.npy` - Training features
- `outputs/data/X_test.npy` - Test features
- `outputs/data/y_train.npy` - Training labels
- `outputs/data/y_test.npy` - Test labels
- `outputs/data/feature_names.txt` - Selected feature names

#### Step 2: Federated Learning Training

```bash
python federated_learning.py
```

This executes **Algorithm 2** (Federated SVM Training) with 5 workers.

**Output**: Console results (F1-score, confusion matrix)

#### Step 3: Generate Figures

```bash
python visualization.py
```

This generates all publication figures:
- Figure 3: Performance comparison
- Figure 4: Model comparison
- Figure 5: Confusion matrix
- Figure 6: Scalability analysis

**Output**: `outputs/figures/*.png` (300 DPI, publication-quality)

---

## Reproducing Paper Results

### Expected Results (Paper Validation)

#### Table 2: Federated vs Centralized Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Centralized SVM | 0.9884 | 0.9965 | 0.9798 | 0.9881 |
| Fed. SVM (3) | 0.9884 | 0.9965 | 0.9798 | 0.9880 |
| Fed. SVM (5) | 0.9884 | 0.9965 | 0.9798 | 0.9881 |
| Fed. SVM (10) | 0.9884 | 0.9965 | 0.9798 | 0.9881 |

**Key Finding**: Performance parity achieved (federated = centralized)

#### Table 3: Algorithm Comparison (5 Workers)

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Federated SVM | 0.9884 | 0.9965 | 0.9798 | 0.9881 |
| Random Forest | 0.9995 | 0.9999 | 0.9990 | 0.9995 |
| ANN | 0.9983 | 0.9997 | 0.9967 | 0.9982 |
| Isolation Forest | 0.4514 | 0.4351 | 0.4008 | 0.4173 |

**Key Finding**: SVM optimal balance (performance vs. resource efficiency)

#### Table 4: Confusion Matrix (100,000 Test Samples)

|             | Predicted Benign | Predicted DDoS |
|-------------|------------------|----------------|
| Actual Benign | 50,829 (TN) | 171 (FP) |
| Actual DDoS | 990 (FN) | 48,010 (TP) |

**Derived Metrics**:
- False Positive Rate: 0.34%
- False Negative Rate: 2.02%
- Specificity: 99.66%

### Validation Checklist

- [ ] Feature reduction: 46 → 17 features (63% reduction)
- [ ] Federated F1-score: 0.9881
- [ ] Performance parity: Fed. (5) = Centralized
- [ ] FPR < 0.5%
- [ ] Recall > 97%

---

## Module Documentation

### 1. config.py

**Purpose**: Centralized configuration management

**Key Parameters**:
- `RANDOM_SEED = 42` - Reproducibility
- `PEARSON_THRESHOLD = 0.2` - Feature selection threshold
- `FEDERATED_ROUNDS = 3` - Training rounds
- `WORKER_COUNTS = [3, 5, 10]` - Network sizes

**Usage**:
```python
import config
config.set_all_seeds()  # Set random seeds
print(config.BENIGN_RATIO)  # Access parameters
```

### 2. data_preprocessing.py

**Purpose**: Data loading and Algorithm 1 implementation

**Classes**:
- `DataLoader`: Loads CIC-IoT2023 or generates synthetic data
- `FeatureSelector`: Implements hybrid Pearson ∩ LDA selection

**Functions**:
- `prepare_data()` - Main entry point

**Usage**:
```python
from data_preprocessing import prepare_data
X_train, X_test, y_train, y_test, features = prepare_data()
```

### 3. federated_learning.py

**Purpose**: Algorithm 2 implementation + comparative models

**Classes**:
- `FederatedSVM`: Federated learning implementation
- `CentralizedModels`: Baseline and comparative models

**Functions**:
- `evaluate_model()` - Comprehensive evaluation

**Usage**:
```python
from federated_learning import FederatedSVM, evaluate_model

# Train federated model
fed_svm = FederatedSVM(n_workers=5, n_rounds=3)
fed_svm.fit(X_train, y_train)

# Evaluate
results = evaluate_model(fed_svm, X_test, y_test, "Fed SVM")
print(f"F1-Score: {results['f1_score']:.4f}")
```

### 4. main_experiment.py

**Purpose**: Orchestrate complete experimental pipeline

**Functions**:
- `run_complete_experiment()` - Execute all phases

**Phases**:
1. Data Preparation
2. Centralized Baseline
3. Federated SVM (multi-size)
4. Comparative Models
5. Results Summary
6. Save Results

**Usage**:
```bash
python main_experiment.py
```

### 5. visualization.py

**Purpose**: Generate publication-quality figures

**Functions**:
- `plot_performance_comparison()` - Figure 3
- `plot_model_comparison()` - Figure 4
- `plot_confusion_matrix()` - Figure 5
- `plot_scalability_analysis()` - Figure 6
- `generate_all_figures()` - Generate all

**Usage**:
```python
from visualization import generate_all_figures
generate_all_figures()
```

---

## Output Files

### JSON Results (`outputs/results/experiment_results.json`)

```json
{
  "centralized_svm": {
    "accuracy": 0.9884,
    "precision": 0.9965,
    "recall": 0.9798,
    "f1_score": 0.9881,
    "training_time": 1.34
  },
  "federated_svm": {
    "3": {...},
    "5": {...},
    "10": {...}
  },
  "comparative_models": {
    "Random Forest": {...},
    "ANN": {...},
    "Isolation Forest": {...}
  }
}
```

### CSV Tables

1. **table2_federated_vs_centralized.csv**: Performance comparison
2. **table3_algorithm_comparison.csv**: Model comparison

### Figures (PNG, 300 DPI)

1. **figure3_performance_comparison.png**: Metrics bar chart
2. **figure4_model_comparison.png**: Algorithm comparison
3. **figure5_confusion_matrix.png**: Confusion matrix heatmap
4. **figure6_scalability_analysis.png**: Scalability analysis

---

## Hardware Requirements

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 20GB
- **OS**: Linux, macOS, or Windows 10/11

### Recommended Requirements (Real Dataset)

- **CPU**: 8+ cores (AMD Ryzen 7/9 or Intel i7/i9)
- **RAM**: 16GB
- **Storage**: 50GB SSD
- **OS**: Ubuntu 22.04 LTS or similar

### GPU

**Not required** - All experiments run on CPU only (paper specification)

---

## Troubleshooting

### Issue 1: Dataset Not Found

```
Error: No CSV files found in ./data/cic-iot-2023
```

**Solution**:
1. Verify dataset path in `config.py`
2. Ensure CSV files are extracted properly
3. OR switch to synthetic mode: `USE_REAL_DATASET = False`

### Issue 2: Memory Error

```
MemoryError: Unable to allocate array
```

**Solution**:
1. Reduce sample size in `config.py`: `REAL_DATASET_SAMPLES = 100000`
2. Close other applications
3. Use synthetic mode for testing

### Issue 3: Module Not Found

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Issue 4: Results Don't Match Paper

**Possible Causes**:
1. Using synthetic data (not real CIC-IoT2023)
2. Different random seed
3. Modified configuration parameters

**Solution**:
1. Ensure `USE_REAL_DATASET = True`
2. Verify `RANDOM_SEED = 42`
3. Check all `config.py` parameters match paper specification

### Issue 5: Visualization Fails

```
FileNotFoundError: Results file not found
```

**Solution**:
```bash
# Run experiment first
python main_experiment.py

# Then generate figures
python visualization.py
```

---

## For Journal Reviewers

### Reproducibility Package

This codebase provides complete reproducibility of all paper results:

1. **Algorithms**: Both Algorithm 1 and Algorithm 2 fully implemented
2. **Tables**: All data tables (2-4) auto-generated as CSV
3. **Figures**: All figures (3-6) auto-generated as PNG
4. **Metrics**: Complete evaluation metrics with validation

### Running Complete Reproduction

```bash
# Single command for complete reproduction
python main_experiment.py

# Generate figures
python visualization.py
```

**Expected Runtime**: ~30-60 minutes (real data, modern CPU)

### Verification Steps

1. Check `outputs/results/table2_federated_vs_centralized.csv`
2. Verify F1-score ≈ 0.9881 for Federated SVM (5 workers)
3. Confirm feature reduction: 46 → 17 features (console output)
4. Review `outputs/figures/` for publication-quality images

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mohamed2026federated,
  title={Privacy-Preserving Federated Learning Framework for DDoS Attack Detection in IoT Networks Using Hybrid Feature Selection},
  author={Mohamed, Haris C P},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```

---

## License

This code is provided for academic research and journal review purposes.

---

## Contact

**Author**: Haris Mohamed C P  
**Email**: harismohamedcp@gmail.com  
**Affiliation**: Department of Computer Science and Engineering, VIT Vellore

For questions about:
- **Code**: Open an issue or email author
- **Dataset**: Contact CIC at https://www.unb.ca/cic/datasets/
- **Paper**: Email author

---

## Acknowledgments

- **CIC-IoT2023 Dataset**: Canadian Institute for Cybersecurity, University of New Brunswick
- **VIT Vellore**: Research and Development Support
- **scikit-learn**: Machine learning library


---

**Last Updated**: March 27, 2026  
**Version**: 1.0.0  
**Status**: Ready for journal submission ✅
