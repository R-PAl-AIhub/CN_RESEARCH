"""
Main Experiment Runner
========================

This script orchestrates the complete experimental pipeline:
1. Data loading and preprocessing (Algorithm 1)
2. Centralized baseline training
3. Federated SVM training across multiple network sizes (Algorithm 2)
4. Comparative evaluation (RF, ANN, IF)
5. Results collection and visualization

Author: Haris Mohamed C P
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path

import config
from data_preprocessing import prepare_data
from federated_learning import FederatedSVM, CentralizedModels, evaluate_model

def run_complete_experiment():
    """
    Execute complete experimental pipeline matching paper methodology
    """
    print("="*80)
    print("FEDERATED LEARNING FOR DDOS DETECTION - COMPLETE EXPERIMENT")
    print("="*80)
    print(f"\nPaper: Privacy-Preserving Federated Learning Framework")
    print(f"Author: Haris Mohamed C P")
    print(f"Institution: VIT Vellore")
    print("="*80)
    
    # Set random seed for reproducibility
    config.set_all_seeds(config.RANDOM_SEED)
    
    # ========================================================================
    # PHASE 1: Data Preparation
    # ========================================================================
    print("\n" + "█"*80)
    print("PHASE 1: DATA PREPARATION")
    print("█"*80)
    
    # Check if preprocessed data exists
    data_files_exist = all([
        (config.DATA_DIR / 'X_train.npy').exists(),
        (config.DATA_DIR / 'X_test.npy').exists(),
        (config.DATA_DIR / 'y_train.npy').exists(),
        (config.DATA_DIR / 'y_test.npy').exists()
    ])
    
    if data_files_exist:
        print("\n✓ Found existing preprocessed data")
        print("Loading from disk...")
        
        X_train = pd.DataFrame(np.load(config.DATA_DIR / 'X_train.npy'))
        X_test = pd.DataFrame(np.load(config.DATA_DIR / 'X_test.npy'))
        y_train = np.load(config.DATA_DIR / 'y_train.npy')
        y_test = np.load(config.DATA_DIR / 'y_test.npy')
        
        with open(config.DATA_DIR / 'feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        
        X_train.columns = feature_names
        X_test.columns = feature_names
        
        print(f"✓ Data loaded: {len(X_train):,} train, {len(X_test):,} test")
    else:
        print("\n⚠ Preprocessed data not found. Running preprocessing...")
        X_train, X_test, y_train, y_test, feature_names = prepare_data()
    
    print(f"\n✓ Features: {len(feature_names)}")
    print(f"✓ Training samples: {len(X_train):,}")
    print(f"✓ Test samples: {len(X_test):,}")
    
    # ========================================================================
    # PHASE 2: Centralized Baseline
    # ========================================================================
    print("\n" + "█"*80)
    print("PHASE 2: CENTRALIZED SVM BASELINE")
    print("█"*80)
    
    centralized_svm, centralized_time = CentralizedModels.train_centralized_svm(
        X_train, y_train
    )
    centralized_results = evaluate_model(
        centralized_svm, X_test, y_test, 
        "Centralized SVM"
    )
    centralized_results['training_time'] = centralized_time
    
    # ========================================================================
    # PHASE 3: Federated SVM Training (Multiple Network Sizes)
    # ========================================================================
    print("\n" + "█"*80)
    print("PHASE 3: FEDERATED SVM TRAINING")
    print("█"*80)
    
    federated_results = {}
    
    for n_workers in config.WORKER_COUNTS:
        print(f"\n{'━'*80}")
        print(f"Training with {n_workers} Workers")
        print(f"{'━'*80}")
        
        start_time = time.time()
        
        # Train federated model
        fed_svm = FederatedSVM(
            n_workers=n_workers,
            n_rounds=config.FEDERATED_ROUNDS
        )
        fed_svm.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Evaluate
        results = evaluate_model(
            fed_svm, X_test, y_test,
            f"Federated SVM ({n_workers} workers)"
        )
        results['training_time'] = training_time
        results['n_workers'] = n_workers
        
        federated_results[n_workers] = results
    
    # ========================================================================
    # PHASE 4: Comparative Models (5 Workers)
    # ========================================================================
    print("\n" + "█"*80)
    print("PHASE 4: COMPARATIVE MODELS (5 Workers Configuration)")
    print("█"*80)
    
    comparative_results = {}
    
    # Random Forest
    print("\n[1/3] Training Random Forest...")
    rf_model, rf_time = CentralizedModels.train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_results['training_time'] = rf_time
    comparative_results['Random Forest'] = rf_results
    
    # Artificial Neural Network
    print("\n[2/3] Training Artificial Neural Network...")
    ann_model, ann_time = CentralizedModels.train_ann(X_train, y_train)
    ann_results = evaluate_model(ann_model, X_test, y_test, "ANN")
    ann_results['training_time'] = ann_time
    comparative_results['ANN'] = ann_results
    
    # Isolation Forest
    print("\n[3/3] Training Isolation Forest...")
    if_model, if_time = CentralizedModels.train_isolation_forest(X_train, y_train)
    if_results = evaluate_model(if_model, X_test, y_test, "Isolation Forest")
    if_results['training_time'] = if_time
    comparative_results['Isolation Forest'] = if_results
    
    # ========================================================================
    # PHASE 5: Results Summary
    # ========================================================================
    print("\n" + "█"*80)
    print("PHASE 5: RESULTS SUMMARY")
    print("█"*80)
    
    # Table 2: Federated vs Centralized Performance
    print("\n" + "─"*80)
    print("Table 2: Federated vs Centralized SVM Performance")
    print("─"*80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Time(s)':>10}")
    print("─"*80)
    
    print(f"{'Centralized SVM':<25} "
          f"{centralized_results['accuracy']:>10.4f} "
          f"{centralized_results['precision']:>10.4f} "
          f"{centralized_results['recall']:>10.4f} "
          f"{centralized_results['f1_score']:>10.4f} "
          f"{centralized_results['training_time']:>10.2f}")
    
    for n_workers in config.WORKER_COUNTS:
        results = federated_results[n_workers]
        print(f"{'Fed. SVM ('+str(n_workers)+' workers)':<25} "
              f"{results['accuracy']:>10.4f} "
              f"{results['precision']:>10.4f} "
              f"{results['recall']:>10.4f} "
              f"{results['f1_score']:>10.4f} "
              f"{results['training_time']:>10.2f}")
    
    # Table 3: Model Comparison (5 Workers)
    print("\n" + "─"*80)
    print("Table 3: Algorithm Comparison (5 Workers, Federated Setting)")
    print("─"*80)
    print(f"{'Algorithm':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Time(s)':>10}")
    print("─"*80)
    
    results_5 = federated_results[5]
    print(f"{'Federated SVM':<20} "
          f"{results_5['accuracy']:>10.4f} "
          f"{results_5['precision']:>10.4f} "
          f"{results_5['recall']:>10.4f} "
          f"{results_5['f1_score']:>10.4f} "
          f"{results_5['training_time']:>10.2f}")
    
    for model_name, results in comparative_results.items():
        print(f"{model_name:<20} "
              f"{results['accuracy']:>10.4f} "
              f"{results['precision']:>10.4f} "
              f"{results['recall']:>10.4f} "
              f"{results['f1_score']:>10.4f} "
              f"{results['training_time']:>10.2f}")
    
    # Table 4: Confusion Matrix (Federated SVM, 5 Workers)
    print("\n" + "─"*80)
    print("Table 4: Confusion Matrix - Federated SVM (5 Workers)")
    print("─"*80)
    results_5 = federated_results[5]
    print(f"                Predicted Benign    Predicted DDoS")
    print(f"Actual Benign   {results_5['tn']:>15,}   {results_5['fp']:>15,}")
    print(f"Actual DDoS     {results_5['fn']:>15,}   {results_5['tp']:>15,}")
    print(f"\nFalse Positive Rate: {results_5['fpr']:.4f} ({results_5['fpr']*100:.2f}%)")
    print(f"False Negative Rate: {results_5['fnr']:.4f} ({results_5['fnr']*100:.2f}%)")
    
    # ========================================================================
    # PHASE 6: Save Results
    # ========================================================================
    print("\n" + "█"*80)
    print("PHASE 6: SAVING RESULTS")
    print("█"*80)
    
    # Prepare results for saving
    all_results = {
        'centralized_svm': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in centralized_results.items() 
                           if k != 'confusion_matrix'},
        'federated_svm': {
            str(n_workers): {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in results.items() 
                            if k != 'confusion_matrix'}
            for n_workers, results in federated_results.items()
        },
        'comparative_models': {
            model_name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in results.items() 
                        if k != 'confusion_matrix'}
            for model_name, results in comparative_results.items()
        },
        'experiment_config': {
            'random_seed': config.RANDOM_SEED,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(feature_names),
            'federated_rounds': config.FEDERATED_ROUNDS,
            'worker_counts': config.WORKER_COUNTS
        }
    }
    
    # Save as JSON
    results_file = config.RESULTS_DIR / 'experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to: {results_file}")
    
    # Save performance comparison table
    performance_table = []
    performance_table.append({
        'Model': 'Centralized SVM',
        'Accuracy': centralized_results['accuracy'],
        'Precision': centralized_results['precision'],
        'Recall': centralized_results['recall'],
        'F1-Score': centralized_results['f1_score'],
        'Time(s)': centralized_results['training_time']
    })
    
    for n_workers in config.WORKER_COUNTS:
        results = federated_results[n_workers]
        performance_table.append({
            'Model': f'Fed. SVM ({n_workers})',
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'Time(s)': results['training_time']
        })
    
    pd.DataFrame(performance_table).to_csv(
        config.RESULTS_DIR / 'table2_federated_vs_centralized.csv',
        index=False
    )
    print(f"✓ Table 2 saved to: table2_federated_vs_centralized.csv")
    
    # Save comparative models table
    comparison_table = []
    comparison_table.append({
        'Algorithm': 'Federated SVM',
        'Accuracy': federated_results[5]['accuracy'],
        'Precision': federated_results[5]['precision'],
        'Recall': federated_results[5]['recall'],
        'F1-Score': federated_results[5]['f1_score'],
        'Time(s)': federated_results[5]['training_time']
    })
    
    for model_name, results in comparative_results.items():
        comparison_table.append({
            'Algorithm': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'Time(s)': results['training_time']
        })
    
    pd.DataFrame(comparison_table).to_csv(
        config.RESULTS_DIR / 'table3_algorithm_comparison.csv',
        index=False
    )
    print(f"✓ Table 3 saved to: table3_algorithm_comparison.csv")
    
    # ========================================================================
    # EXPERIMENT COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\n✓ All results saved to: {config.RESULTS_DIR}")
    print(f"✓ Check experiment_results.json for complete results")
    print(f"✓ CSV tables ready for paper submission")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    try:
        results = run_complete_experiment()
        print("\n✓ SUCCESS: Experiment completed without errors")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
