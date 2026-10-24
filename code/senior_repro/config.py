"""
Configuration Module for Federated Learning DDoS Detection
============================================================

This module contains all configuration parameters and constants used throughout
the federated learning framework for IoT DDoS detection.

Paper Reference:
"Privacy-Preserving Federated Learning Framework for DDoS Attack Detection 
in IoT Networks Using Hybrid Feature Selection"

Author: Haris Mohamed C P
Affiliation: VIT Vellore
"""

import os
from pathlib import Path

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42  # Used consistently across all experiments

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Set to True to use real CIC-IoT2023 dataset, False for synthetic demo
USE_REAL_DATASET = True

# Path to CIC-IoT2023 dataset (only used if USE_REAL_DATASET = True)
# Adjust this path to your dataset location
DATASET_PATH = "../../data/repro/cic-iot-2023"  # Default path

# Sample sizes as per paper
# Paper specifies: 500,000 samples from ~46 million total records (1% sample)
REAL_DATASET_SAMPLES = 500000   # 500K samples for real data
SYNTHETIC_SAMPLES = 50000       # 50K samples for synthetic demo

# ============================================================================
# CLASS DISTRIBUTION (FROM PAPER)
# ============================================================================
# Paper page 11: "51% Benign, 49% DDoS"
BENIGN_RATIO = 0.51
DDOS_RATIO = 0.49

# ============================================================================
# FEATURE SELECTION PARAMETERS (ALGORITHM 1)
# ============================================================================
# Paper Algorithm 1: Pearson correlation threshold
PEARSON_THRESHOLD = 0.2  # |ρ| > 0.2

# Variance threshold for data cleaning (Step 1)
VARIANCE_THRESHOLD = 0.01  # Remove features with variance < 0.01

# LDA selection: Top N features
LDA_TOP_N = 20  # Paper specifies top-20 LDA features

# ============================================================================
# CROSS-VALIDATION PARAMETERS
# ============================================================================
# Paper: "10-fold stratified cross-validation"
CV_FOLDS = 10

# Cross-validation score range for overfitting check (Algorithm 1)
# Paper Algorithm 1: "if 0.9 < 10-fold cross validation score < 0.97"
MIN_CV_SCORE = 0.90
MAX_CV_SCORE = 0.97

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
# Paper: "80-20 stratified split"
TRAIN_SIZE = 0.80
TEST_SIZE = 0.20

# ============================================================================
# FEDERATED LEARNING CONFIGURATION (ALGORITHM 2)
# ============================================================================
# Number of federated rounds (Paper: R=3)
FEDERATED_ROUNDS = 3

# Network sizes to evaluate (Paper: 3, 5, 10 workers)
WORKER_COUNTS = [3, 5, 10]

# ============================================================================
# SVM HYPERPARAMETERS
# ============================================================================
# Paper Section III: SVM configuration
SVM_C = 1.0                # Regularization parameter
SVM_MAX_ITER = 1000        # Maximum iterations
SVM_KERNEL = 'linear'      # Linear kernel
SVM_RANDOM_STATE = RANDOM_SEED

# ============================================================================
# RANDOM FOREST HYPERPARAMETERS
# ============================================================================
# Paper Table 2: Random Forest parameters
RF_N_ESTIMATORS = 100      # Number of trees
RF_MAX_DEPTH = 10          # Maximum depth
RF_MIN_SAMPLES_SPLIT = 2   # Minimum samples per split
RF_RANDOM_STATE = RANDOM_SEED
RF_N_JOBS = -1             # Use all available cores

# ============================================================================
# ARTIFICIAL NEURAL NETWORK HYPERPARAMETERS
# ============================================================================
# Paper Table 2: ANN architecture (17→100→50→1)
ANN_HIDDEN_LAYERS = (100, 50)  # Hidden layer sizes
ANN_ACTIVATION = 'relu'         # Activation function
ANN_SOLVER = 'adam'             # Optimizer
ANN_ALPHA = 0.001               # Learning rate
ANN_BATCH_SIZE = 32             # Batch size
ANN_MAX_ITER = 10               # Epochs
ANN_RANDOM_STATE = RANDOM_SEED

# ============================================================================
# ISOLATION FOREST HYPERPARAMETERS
# ============================================================================
# Paper Table 2: Isolation Forest parameters
IF_N_ESTIMATORS = 100          # Number of estimators
IF_CONTAMINATION = 0.49        # Expected outlier fraction (matching DDoS ratio)
IF_MAX_SAMPLES = 256           # Max samples per tree
IF_RANDOM_STATE = RANDOM_SEED

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
OUTPUT_DIR = Path("./outputs")
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR, DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Figure size for plots
FIGURE_SIZE = (10, 6)
FIGURE_DPI = 300

# Color palette
COLOR_PALETTE = "husl"

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FILE = OUTPUT_DIR / "experiment.log"

# ============================================================================
# MEMORY AND PERFORMANCE TRACKING
# ============================================================================
TRACK_MEMORY = True     # Track peak memory usage
TRACK_TIME = True       # Track training time

# ============================================================================
# REPRODUCIBILITY SETTINGS
# ============================================================================
# Set all random seeds for reproducibility
def set_all_seeds(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    # If using any deep learning frameworks
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

# ============================================================================
# PAPER-SPECIFIC CONSTANTS FOR VALIDATION
# ============================================================================
# These values are from the paper and used for validation

# Expected feature reduction (Paper: 63% reduction, 46→17 features)
EXPECTED_ORIGINAL_FEATURES = 46
EXPECTED_FINAL_FEATURES = 17
EXPECTED_REDUCTION_PERCENT = 63.0

# Expected performance metrics (Paper Results)
EXPECTED_CENTRALIZED_ACCURACY = 0.9884
EXPECTED_CENTRALIZED_PRECISION = 0.9965
EXPECTED_CENTRALIZED_RECALL = 0.9798
EXPECTED_CENTRALIZED_F1 = 0.9881

# Expected federated performance (5 workers)
EXPECTED_FEDERATED_ACCURACY = 0.9884
EXPECTED_FEDERATED_F1 = 0.9881

# Expected confusion matrix values (100,000 test samples)
EXPECTED_TRUE_NEGATIVES = 50829
EXPECTED_FALSE_POSITIVES = 171
EXPECTED_FALSE_NEGATIVES = 990
EXPECTED_TRUE_POSITIVES = 48010

# Expected FPR and FNR
EXPECTED_FPR = 0.0034  # 0.34%
EXPECTED_FNR = 0.0202  # 2.02%

if __name__ == "__main__":
    print("="*80)
    print("CONFIGURATION MODULE")
    print("="*80)
    print(f"\nRandom Seed: {RANDOM_SEED}")
    print(f"Use Real Dataset: {USE_REAL_DATASET}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Target Samples: {REAL_DATASET_SAMPLES:,}")
    print(f"\nClass Distribution:")
    print(f"  Benign: {BENIGN_RATIO*100:.1f}%")
    print(f"  DDoS:   {DDOS_RATIO*100:.1f}%")
    print(f"\nFeature Selection:")
    print(f"  Pearson Threshold: {PEARSON_THRESHOLD}")
    print(f"  LDA Top-N: {LDA_TOP_N}")
    print(f"\nFederated Learning:")
    print(f"  Rounds: {FEDERATED_ROUNDS}")
    print(f"  Worker Counts: {WORKER_COUNTS}")
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print("\n✓ Configuration loaded successfully")
