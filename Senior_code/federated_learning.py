"""
Federated Learning Module
===========================

Implementation of Algorithm 2: Federated SVM Training

This module implements the federated learning framework for privacy-preserving
DDoS detection across distributed IoT networks.

Paper Reference: Algorithm 2 - Federated SVM Training

Author: Haris Mohamed C P
"""

import numpy as np
import pandas as pd
import time
from copy import deepcopy
import gc

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix)

import config


class FederatedSVM:
    """
    Federated Support Vector Machine implementation
    Algorithm 2 from the paper
    """
    
    def __init__(self, n_workers=5, n_rounds=3):
        """
        Initialize Federated SVM
        
        Args:
            n_workers: Number of federated workers (Paper: 3, 5, 10)
            n_rounds: Number of federated rounds (Paper: R=3)
        """
        self.n_workers = n_workers
        self.n_rounds = n_rounds
        self.global_model = None
        self.worker_models = []
        self.training_history = []
        
        print(f"Initialized Federated SVM:")
        print(f"  Workers: {n_workers}")
        print(f"  Rounds: {n_rounds}")
    
    def fit(self, X_train, y_train):
        """
        Train federated SVM using Algorithm 2
        
        Algorithm 2 Steps:
        1. Partition data into N workers
        2. For each round:
           - Local training (parallel)
           - Parameter aggregation (central)
           - Global model update
        """
        print("\n" + "="*80)
        print(f"ALGORITHM 2: FEDERATED SVM TRAINING ({self.n_workers} workers)")
        print("="*80)
        
        start_time = time.time()
        
        # Partition data
        print(f"\n[Initialization] Partitioning data into {self.n_workers} workers...")
        data_partitions = self._partition_data(X_train, y_train)
        
        for worker_id, (X_local, y_local) in enumerate(data_partitions):
            print(f"  Worker {worker_id+1}: {len(X_local):,} samples "
                  f"({(y_local==0).sum():,} benign, {(y_local==1).sum():,} DDoS)")
        
        # Initialize global model
        self.global_model = LinearSVC(
            C=config.SVM_C,
            max_iter=config.SVM_MAX_ITER,
            random_state=config.RANDOM_SEED
        )
        
        # Federated training rounds
        for round_num in range(self.n_rounds):
            print(f"\n{'─'*80}")
            print(f"ROUND {round_num + 1}/{self.n_rounds}")
            print(f"{'─'*80}")
            
            # Local training phase (parallel in simulation)
            print(f"\n[Phase 1] Local Training (Parallel)...")
            worker_params = []
            worker_sizes = []
            
            for worker_id, (X_local, y_local) in enumerate(data_partitions):
                # Train local model
                local_model = LinearSVC(
                    C=config.SVM_C,
                    max_iter=config.SVM_MAX_ITER,
                    random_state=config.RANDOM_SEED
                )
                
                local_model.fit(X_local, y_local)
                
                # Extract parameters (weights, bias)
                w_i = local_model.coef_[0]
                b_i = local_model.intercept_[0]
                n_i = len(X_local)
                
                worker_params.append((w_i, b_i))
                worker_sizes.append(n_i)
                
                # Local evaluation
                y_pred_local = local_model.predict(X_local)
                local_f1 = f1_score(y_local, y_pred_local)
                
                print(f"  Worker {worker_id+1}: F1={local_f1:.4f}, n={n_i:,}")
            
            # Aggregation phase (central server)
            print(f"\n[Phase 2] Aggregation (Central Server)...")
            w_global, b_global = self._aggregate_parameters(worker_params, worker_sizes)
            
            # Update global model
            self.global_model.coef_ = np.array([w_global])
            self.global_model.intercept_ = np.array([b_global])
            self.global_model.classes_ = np.array([0, 1])
            
            # Communication overhead
            bytes_per_worker = 18 * 4  # 18 floats (17 weights + 1 bias) * 4 bytes
            total_bytes = bytes_per_worker * self.n_workers
            print(f"  Communication: {total_bytes} bytes ({total_bytes/1024:.2f} KB)")
            
            # Track history
            self.training_history.append({
                'round': round_num + 1,
                'worker_params': deepcopy(worker_params),
                'global_params': (deepcopy(w_global), deepcopy(b_global))
            })
        
        total_time = time.time() - start_time
        
        print(f"\n✓ Federated training complete")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per round: {total_time/self.n_rounds:.2f}s")
        
        return self
    
    def _partition_data(self, X, y):
        """Partition data into N equal parts (IID partitioning)"""
        n_samples = len(X)
        partition_size = n_samples // self.n_workers
        
        # Shuffle indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        partitions = []
        for i in range(self.n_workers):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < self.n_workers - 1 else n_samples
            
            partition_indices = indices[start_idx:end_idx]
            X_partition = X.iloc[partition_indices] if isinstance(X, pd.DataFrame) else X[partition_indices]
            y_partition = y.iloc[partition_indices] if isinstance(y, pd.Series) else y[partition_indices]
            
            partitions.append((X_partition, y_partition))
        
        return partitions
    
    def _aggregate_parameters(self, worker_params, worker_sizes):
        """
        Weighted parameter aggregation
        
        Paper Equation: w_global = Σ(n_i/n_total × w_i)
        """
        n_total = sum(worker_sizes)
        
        # Initialize aggregated parameters
        n_features = len(worker_params[0][0])
        w_global = np.zeros(n_features)
        b_global = 0.0
        
        # Weighted averaging
        for (w_i, b_i), n_i in zip(worker_params, worker_sizes):
            weight = n_i / n_total
            w_global += weight * w_i
            b_global += weight * b_i
        
        return w_global, b_global
    
    def predict(self, X):
        """Predict using global model"""
        if self.global_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.global_model.predict(X)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class CentralizedModels:
    """Centralized baseline models for comparison"""
    
    @staticmethod
    def train_centralized_svm(X_train, y_train):
        """Train centralized SVM (privacy-violating baseline)"""
        print("\n" + "="*80)
        print("CENTRALIZED SVM (Privacy-Violating Baseline)")
        print("="*80)
        
        start_time = time.time()
        
        model = LinearSVC(
            C=config.SVM_C,
            max_iter=config.SVM_MAX_ITER,
            random_state=config.RANDOM_SEED
        )
        
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        print(f"✓ Training complete in {train_time:.2f}s")
        
        return model, train_time
    
    @staticmethod
    def train_random_forest(X_train, y_train):
        """Train Random Forest (Paper Table 2 parameters)"""
        print("\n" + "="*80)
        print("RANDOM FOREST CLASSIFIER")
        print("="*80)
        
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=config.RF_N_JOBS
        )
        
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        print(f"✓ Training complete in {train_time:.2f}s")
        
        return model, train_time
    
    @staticmethod
    def train_ann(X_train, y_train):
        """Train Artificial Neural Network (Paper Table 2: 17→100→50→1)"""
        print("\n" + "="*80)
        print("ARTIFICIAL NEURAL NETWORK")
        print("="*80)
        
        start_time = time.time()
        
        model = MLPClassifier(
            hidden_layer_sizes=config.ANN_HIDDEN_LAYERS,
            activation=config.ANN_ACTIVATION,
            solver=config.ANN_SOLVER,
            alpha=config.ANN_ALPHA,
            batch_size=config.ANN_BATCH_SIZE,
            max_iter=config.ANN_MAX_ITER,
            random_state=config.ANN_RANDOM_STATE
        )
        
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        print(f"✓ Training complete in {train_time:.2f}s")
        
        return model, train_time
    
    @staticmethod
    def train_isolation_forest(X_train, y_train):
        """Train Isolation Forest (Paper Table 2 parameters)"""
        print("\n" + "="*80)
        print("ISOLATION FOREST")
        print("="*80)
        
        start_time = time.time()
        
        model = IsolationForest(
            n_estimators=config.IF_N_ESTIMATORS,
            contamination=config.IF_CONTAMINATION,
            max_samples=config.IF_MAX_SAMPLES,
            random_state=config.IF_RANDOM_STATE
        )
        
        model.fit(X_train)
        
        train_time = time.time() - start_time
        
        print(f"✓ Training complete in {train_time:.2f}s")
        
        return model, train_time


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation
    
    Returns metrics as per paper:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - FPR, FNR
    """
    print(f"\n{'─'*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'─'*80}")
    
    # Predictions
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Model must have predict method")
    
    # Handle Isolation Forest (returns -1/1 instead of 0/1)
    if isinstance(model, IsolationForest):
        y_pred = (y_pred == -1).astype(int)  # -1 (outlier) → 1 (attack), 1 (normal) → 0 (benign)
    
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Derived metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,}  |  FP: {fp:,}")
    print(f"  FN: {fn:,}  |  TP: {tp:,}")
    
    print(f"\nError Rates:")
    print(f"  FPR (False Positive Rate): {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  FNR (False Negative Rate): {fnr:.4f} ({fnr*100:.2f}%)")
    print(f"  Specificity: {specificity:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'fpr': fpr, 'fnr': fnr, 'specificity': specificity
    }


if __name__ == "__main__":
    print("="*80)
    print("FEDERATED LEARNING MODULE")
    print("="*80)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train = pd.DataFrame(np.load(config.DATA_DIR / 'X_train.npy'))
    X_test = pd.DataFrame(np.load(config.DATA_DIR / 'X_test.npy'))
    y_train = np.load(config.DATA_DIR / 'y_train.npy')
    y_test = np.load(config.DATA_DIR / 'y_test.npy')
    
    print(f"✓ Data loaded: {len(X_train):,} train, {len(X_test):,} test")
    
    # Train Federated SVM (5 workers)
    fed_svm = FederatedSVM(n_workers=5, n_rounds=3)
    fed_svm.fit(X_train, y_train)
    
    # Evaluate
    results = evaluate_model(fed_svm, X_test, y_test, "Federated SVM (5 workers)")
    
    print(f"\n✓ F1-Score: {results['f1_score']:.4f}")
