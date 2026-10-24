"""
Data Loading and Preprocessing Module
======================================

This module handles:
1. Loading CIC-IoT2023 dataset or generating synthetic data
2. Data cleaning and preprocessing
3. Feature selection (Pearson + LDA intersection)
4. Train-test splitting

Paper Reference: Algorithm 1 - Hybrid Feature Selection (Pearson ∩ LDA)

Author: Haris Mohamed C P
"""

import numpy as np
import pandas as pd
from pathlib import Path
import gc
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC

import config

class DataLoader:
    """Load and preprocess CIC-IoT2023 dataset or generate synthetic data"""
    
    def __init__(self, use_real_dataset=config.USE_REAL_DATASET):
        self.use_real_dataset = use_real_dataset
        self.data = None
        self.label_column = 'label'
        
    def load_data(self):
        """Load dataset based on configuration"""
        if self.use_real_dataset:
            print("\n" + "="*80)
            print("LOADING REAL CIC-IoT2023 DATASET")
            print("="*80)
            self.data = self._load_cic_iot2023(
                config.DATASET_PATH,
                config.REAL_DATASET_SAMPLES
            )
        else:
            print("\n" + "="*80)
            print("GENERATING SYNTHETIC DATASET (DEMO MODE)")
            print("="*80)
            self.data = self._generate_synthetic_data(config.SYNTHETIC_SAMPLES)
        
        return self.data
    
    def _load_cic_iot2023(self, dataset_path, target_samples=500000):
        """
        Load CIC-IoT2023 dataset with proper sampling
        
        Paper specifies:
        - Original: ~46 million records
        - Binary classification: Benign vs DDoS
        - Balance: 51% Benign, 49% DDoS
        """
        dataset_path = Path(dataset_path)
        csv_files = list(dataset_path.glob('*.csv'))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Calculate targets
        n_benign = int(target_samples * config.BENIGN_RATIO)
        n_ddos = target_samples - n_benign
        
        print(f"\nTarget samples:")
        print(f"  Benign: {n_benign:,} ({config.BENIGN_RATIO*100:.1f}%)")
        print(f"  DDoS:   {n_ddos:,} ({(1-config.BENIGN_RATIO)*100:.1f}%)")
        print(f"  Total:  {target_samples:,}")
        
        benign_samples = []
        ddos_samples = []
        benign_collected = 0
        ddos_collected = 0
        
        # Read files
        for csv_file in csv_files:
            if benign_collected >= n_benign and ddos_collected >= n_ddos:
                break
            
            print(f"\nProcessing: {csv_file.name}")
            
            try:
                for chunk in pd.read_csv(csv_file, chunksize=50000, low_memory=False):
                    # Find label column
                    label_col = None
                    for col in chunk.columns:
                        if col.lower() in ['label', 'attack', 'attack_type', 'class']:
                            label_col = col
                            break
                    
                    if label_col is None:
                        break
                    
                    # Collect Benign
                    if benign_collected < n_benign:
                        benign_mask = chunk[label_col].astype(str).str.contains(
                            'benign|normal', case=False, na=False
                        )
                        benign_chunk = chunk[benign_mask]
                        
                        if len(benign_chunk) > 0:
                            needed = n_benign - benign_collected
                            sample = benign_chunk.sample(
                                n=min(needed, len(benign_chunk)),
                                random_state=config.RANDOM_SEED
                            )
                            benign_samples.append(sample)
                            benign_collected += len(sample)
                    
                    # Collect DDoS
                    if ddos_collected < n_ddos:
                        ddos_mask = chunk[label_col].astype(str).str.contains(
                            'ddos', case=False, na=False
                        )
                        ddos_chunk = chunk[ddos_mask]
                        
                        if len(ddos_chunk) > 0:
                            needed = n_ddos - ddos_collected
                            sample = ddos_chunk.sample(
                                n=min(needed, len(ddos_chunk)),
                                random_state=config.RANDOM_SEED
                            )
                            ddos_samples.append(sample)
                            ddos_collected += len(sample)
                    
                    print(f"  Progress: Benign {benign_collected:,}/{n_benign:,}, "
                          f"DDoS {ddos_collected:,}/{n_ddos:,}", end='\r')
                    
                    if benign_collected >= n_benign and ddos_collected >= n_ddos:
                        break
                    
                    gc.collect()
            
            except Exception as e:
                print(f"\n  Error: {e}")
                continue
        
        print("\n")
        
        # Combine
        if len(benign_samples) == 0 or len(ddos_samples) == 0:
            raise ValueError("Could not find enough samples")
        
        df_benign = pd.concat(benign_samples, ignore_index=True)
        df_ddos = pd.concat(ddos_samples, ignore_index=True)
        
        # Create binary labels
        df_benign['label'] = 0
        df_ddos['label'] = 1
        
        # Combine and shuffle
        df = pd.concat([df_benign, df_ddos], ignore_index=True)
        df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
        
        print(f"✓ Dataset loaded:")
        print(f"  Benign: {len(df_benign):,}")
        print(f"  DDoS:   {len(df_ddos):,}")
        print(f"  Total:  {len(df):,}")
        print(f"  Features: {df.shape[1]}")
        
        return df
    
    def _generate_synthetic_data(self, n_samples=50000):
        """
        Generate synthetic data for demonstration
        Mimics CIC-IoT2023 characteristics
        """
        print(f"Generating {n_samples:,} synthetic samples...")
        
        # Create 46 features as per paper
        n_features = 46
        n_benign = int(n_samples * config.BENIGN_RATIO)
        n_ddos = n_samples - n_benign
        
        # Generate benign traffic (normal distribution)
        benign_data = np.random.normal(loc=0, scale=1, size=(n_benign, n_features))
        benign_labels = np.zeros(n_benign)
        
        # Generate DDoS traffic (shifted distribution with higher variance)
        ddos_data = np.random.normal(loc=2, scale=2, size=(n_ddos, n_features))
        ddos_labels = np.ones(n_ddos)
        
        # Combine
        X = np.vstack([benign_data, ddos_data])
        y = np.concatenate([benign_labels, ddos_labels])
        
        # Shuffle
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y.astype(int)
        
        print(f"✓ Synthetic dataset created:")
        print(f"  Benign: {n_benign:,}")
        print(f"  DDoS:   {n_ddos:,}")
        print(f"  Total:  {n_samples:,}")
        print(f"  Features: {n_features}")
        
        return df


class FeatureSelector:
    """
    Hybrid Feature Selection using Pearson ∩ LDA
    Implementation of Algorithm 1 from the paper
    """
    
    def __init__(self):
        self.selected_features = None
        self.pearson_features = None
        self.lda_features = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, df, label_column='label'):
        """
        Execute Algorithm 1: Hybrid Feature Selection
        
        Steps:
        1. Data Cleaning (variance filtering)
        2. Pearson Correlation Selection (|ρ| > 0.2)
        3. LDA Selection (top-20 features)
        4. Feature Intersection
        """
        print("\n" + "="*80)
        print("ALGORITHM 1: HYBRID FEATURE SELECTION (Pearson ∩ LDA)")
        print("="*80)
        
        # Separate features and labels
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        print(f"\nOriginal features: {X.shape[1]}")
        
        # Step 1: Data Cleaning
        print("\n[Step 1] Data Cleaning...")
        X_clean = self._variance_filter(X)
        print(f"  After variance filter: {X_clean.shape[1]} features")
        
        # Step 2: Pearson Correlation Selection
        print("\n[Step 2] Pearson Correlation Selection...")
        pearson_features = self._pearson_selection(X_clean, y)
        self.pearson_features = pearson_features
        print(f"  Pearson selected: {len(pearson_features)} features")
        
        # Step 3: LDA Selection
        print("\n[Step 3] Linear Discriminant Analysis...")
        lda_features = self._lda_selection(X_clean, y)
        self.lda_features = lda_features
        print(f"  LDA selected: {len(lda_features)} features")
        
        # Step 4: Feature Intersection
        print("\n[Step 4] Feature Intersection...")
        final_features = list(set(pearson_features) & set(lda_features))
        self.selected_features = final_features
        
        reduction = (1 - len(final_features) / X.shape[1]) * 100
        
        print(f"  Final features: {len(final_features)}")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"  Features: {final_features[:5]}..." if len(final_features) > 5 else f"  Features: {final_features}")
        
        # Create reduced dataset
        X_reduced = X_clean[final_features]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_reduced),
            columns=final_features
        )
        
        # Cross-validation check (Algorithm 1)
        print("\n[Step 5] Cross-Validation Check...")
        cv_score = self._cross_validation_check(X_scaled, y)
        
        if config.MIN_CV_SCORE < cv_score < config.MAX_CV_SCORE:
            print(f"  ✓ CV score {cv_score:.4f} within acceptable range "
                  f"({config.MIN_CV_SCORE}, {config.MAX_CV_SCORE})")
        else:
            print(f"  ⚠ CV score {cv_score:.4f} outside expected range")
        
        return X_scaled, y
    
    def _variance_filter(self, X, threshold=config.VARIANCE_THRESHOLD):
        """Step 1: Remove features with variance < threshold"""
        variances = X.var()
        keep_features = variances[variances >= threshold].index.tolist()
        return X[keep_features]
    
    def _pearson_selection(self, X, y, threshold=config.PEARSON_THRESHOLD):
        """Step 2: Select features with |ρ| > threshold"""
        correlations = X.corrwith(pd.Series(y)).abs()
        selected = correlations[correlations > threshold].index.tolist()
        return selected
    
    def _lda_selection(self, X, y, top_n=config.LDA_TOP_N):
        """Step 3: Select top-N LDA features"""
        # Standardize for LDA
        X_scaled = StandardScaler().fit_transform(X)
        
        # Fit LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_scaled, y)
        
        # Get feature importances
        # For binary LDA, use absolute values of coefficients
        importances = np.abs(lda.coef_[0])
        
        # Select top-N
        top_indices = np.argsort(importances)[::-1][:top_n]
        selected = X.columns[top_indices].tolist()
        
        return selected
    
    def _cross_validation_check(self, X, y):
        """Step 5: Cross-validation for overfitting check"""
        svm = LinearSVC(C=config.SVM_C, max_iter=config.SVM_MAX_ITER, 
                       random_state=config.RANDOM_SEED)
        
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, 
                            random_state=config.RANDOM_SEED)
        
        scores = cross_val_score(svm, X, y, cv=cv, scoring='f1', n_jobs=-1)
        mean_score = scores.mean()
        
        print(f"  10-fold CV scores: {scores}")
        print(f"  Mean F1: {mean_score:.4f} (σ={scores.std():.4f})")
        
        return mean_score
    
    def transform(self, X):
        """Transform new data using selected features"""
        if self.selected_features is None:
            raise ValueError("Must call fit_transform first")
        
        X_selected = X[self.selected_features]
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_selected),
            columns=self.selected_features
        )
        return X_scaled


def prepare_data():
    """
    Main function to load and preprocess data
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    # Load data
    loader = DataLoader(use_real_dataset=config.USE_REAL_DATASET)
    df = loader.load_data()
    
    # Feature selection
    selector = FeatureSelector()
    X, y = selector.fit_transform(df)
    
    # Train-test split (80-20 stratified)
    print("\n" + "="*80)
    print("TRAIN-TEST SPLIT (80-20 Stratified)")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED
    )
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"  Benign: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
    print(f"  DDoS:   {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")
    
    print(f"\nTest set: {len(X_test):,} samples")
    print(f"  Benign: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.1f}%)")
    print(f"  DDoS:   {(y_test==1).sum():,} ({(y_test==1).mean()*100:.1f}%)")
    
    # Save preprocessed data
    print(f"\n✓ Saving preprocessed data to {config.DATA_DIR}")
    np.save(config.DATA_DIR / 'X_train.npy', X_train.values)
    np.save(config.DATA_DIR / 'X_test.npy', X_test.values)
    np.save(config.DATA_DIR / 'y_train.npy', y_train.values)
    np.save(config.DATA_DIR / 'y_test.npy', y_test.values)
    
    with open(config.DATA_DIR / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


if __name__ == "__main__":
    print("="*80)
    print("DATA PREPROCESSING MODULE")
    print("="*80)
    
    # Set random seed
    config.set_all_seeds()
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"✓ Training samples: {len(X_train):,}")
    print(f"✓ Test samples: {len(X_test):,}")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Data saved to: {config.DATA_DIR}")
