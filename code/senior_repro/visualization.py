"""
Visualization Module
====================

Generate all figures for the paper:
- Figure 1: Federated Architecture (conceptual)
- Figure 2: Feature Selection Flow (conceptual)
- Figure 3: Performance Metrics Comparison (Table 2 visualization)
- Figure 4: Model Comparison Bar Chart (Table 3 visualization)
- Figure 5: Confusion Matrix Heatmap
- Figure 6: Scalability Analysis

Author: Haris Mohamed C P
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

import config

# Set visualization style
plt.style.use(config.PLOT_STYLE)
sns.set_palette(config.COLOR_PALETTE)


def load_results():
    """Load experiment results from JSON"""
    results_file = config.RESULTS_DIR / 'experiment_results.json'
    
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            "Please run main_experiment.py first"
        )
    
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_performance_comparison(results, save=True):
    """
    Figure 3: Performance Metrics Comparison
    Bar chart comparing Centralized vs Federated SVM across network sizes
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    # Prepare data
    models = ['Centralized SVM']
    accuracies = [results['centralized_svm']['accuracy']]
    precisions = [results['centralized_svm']['precision']]
    recalls = [results['centralized_svm']['recall']]
    f1_scores = [results['centralized_svm']['f1_score']]
    
    for n_workers in [3, 5, 10]:
        models.append(f'Fed. SVM ({n_workers})')
        fed_results = results['federated_svm'][str(n_workers)]
        accuracies.append(fed_results['accuracy'])
        precisions.append(fed_results['precision'])
        recalls.append(fed_results['recall'])
        f1_scores.append(fed_results['f1_score'])
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics: Centralized vs Federated SVM', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0.95, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_file = config.FIGURES_DIR / 'figure3_performance_comparison.png'
        plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Figure 3 saved: {output_file}")
    
    return fig


def plot_model_comparison(results, save=True):
    """
    Figure 4: Model Comparison (5 Workers)
    Bar chart comparing SVM, RF, ANN, IF in federated setting
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    
    # Prepare data
    models = ['SVM (5)', 'RF (5)', 'ANN (5)', 'IF (5)']
    
    # Get results
    svm_results = results['federated_svm']['5']
    rf_results = results['comparative_models']['Random Forest']
    ann_results = results['comparative_models']['ANN']
    if_results = results['comparative_models']['Isolation Forest']
    
    accuracies = [svm_results['accuracy'], rf_results['accuracy'], 
                  ann_results['accuracy'], if_results['accuracy']]
    precisions = [svm_results['precision'], rf_results['precision'],
                  ann_results['precision'], if_results['precision']]
    recalls = [svm_results['recall'], rf_results['recall'],
               ann_results['recall'], if_results['recall']]
    f1_scores = [svm_results['f1_score'], rf_results['f1_score'],
                 ann_results['f1_score'], if_results['f1_score']]
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison (5 Workers, Federated Setting)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_file = config.FIGURES_DIR / 'figure4_model_comparison.png'
        plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Figure 4 saved: {output_file}")
    
    return fig


def plot_confusion_matrix(results, save=True):
    """
    Figure 5: Confusion Matrix Heatmap
    Federated SVM (5 Workers) on test set
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get confusion matrix values
    svm_results = results['federated_svm']['5']
    cm = np.array([[svm_results['tn'], svm_results['fp']],
                   [svm_results['fn'], svm_results['tp']]])
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', 
                xticklabels=['Benign', 'DDoS'],
                yticklabels=['Benign', 'DDoS'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Federated SVM (5 Workers)',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save:
        output_file = config.FIGURES_DIR / 'figure5_confusion_matrix.png'
        plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Figure 5 saved: {output_file}")
    
    return fig


def plot_scalability_analysis(results, save=True):
    """
    Figure 6: Scalability Analysis
    F1-Score and Training Time vs Number of Workers
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data
    worker_counts = [3, 5, 10]
    f1_scores = [results['federated_svm'][str(n)]['f1_score'] 
                 for n in worker_counts]
    train_times = [results['federated_svm'][str(n)]['training_time']
                   for n in worker_counts]
    
    # Centralized baseline
    central_f1 = results['centralized_svm']['f1_score']
    central_time = results['centralized_svm']['training_time']
    
    # Plot 1: F1-Score vs Worker Count
    ax1.plot(worker_counts, f1_scores, 'o-', linewidth=2, markersize=8, 
             label='Federated SVM')
    ax1.axhline(y=central_f1, color='r', linestyle='--', linewidth=2,
                label='Centralized SVM')
    ax1.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score vs Network Size', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0.97, 1.0])
    
    # Plot 2: Training Time vs Worker Count
    ax2.bar(worker_counts, train_times, alpha=0.7, width=1.5)
    ax2.axhline(y=central_time, color='r', linestyle='--', linewidth=2,
                label='Centralized SVM')
    ax2.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time vs Network Size', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_file = config.FIGURES_DIR / 'figure6_scalability_analysis.png'
        plt.savefig(output_file, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Figure 6 saved: {output_file}")
    
    return fig


def generate_all_figures():
    """Generate all figures for the paper"""
    print("="*80)
    print("GENERATING PAPER FIGURES")
    print("="*80)
    
    # Load results
    print("\nLoading experiment results...")
    results = load_results()
    print("✓ Results loaded")
    
    # Generate figures
    print("\nGenerating figures...")
    
    plot_performance_comparison(results)
    plot_model_comparison(results)
    plot_confusion_matrix(results)
    plot_scalability_analysis(results)
    
    print("\n" + "="*80)
    print("ALL FIGURES GENERATED")
    print("="*80)
    print(f"\n✓ Figures saved to: {config.FIGURES_DIR}")
    print("\nGenerated files:")
    for fig_file in sorted(config.FIGURES_DIR.glob('*.png')):
        print(f"  - {fig_file.name}")


if __name__ == "__main__":
    generate_all_figures()
