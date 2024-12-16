"""
Plotting module for visualizing model performance on held-out sets.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
import yaml
from typing import Dict

def load_results(results_file: str) -> Dict:
    """Load results from YAML file."""
    with open(results_file, 'r') as f:
        return yaml.safe_load(f)

def plot_regression_metrics(results: Dict, predictions: Dict, output_dir: str, 
                          model_name: str, dataset_type: str = "extended"):
    """Plot regression metrics with predicted enrichment vs 1/experimental_kd."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get metrics based on dataset type
    metrics = results['all'] if dataset_type == "extended" else results['lib']
    preds = predictions['all'] if dataset_type == "extended" else predictions['lib']
    
    # Convert Kd to 1/Kd for both conditions
    y_true_on_inv = 1/np.log10(preds['on']['true'])
    y_true_off_inv = 1/np.log10(preds['off']['true'])
    
    # Calculate Pearson correlations
    pearson_on = stats.pearsonr(preds['on']['pred'], y_true_on_inv)[0]
    pearson_off = stats.pearsonr(preds['off']['pred'], y_true_off_inv)[0]
    
    # On-DNA plot
    ax1.scatter(y_true_on_inv, preds['on']['pred'], alpha=0.5)
    metrics_text_on = (f'Train MSE: {results["train"]["mse"]:.3f}\n'
                      f'Valid MSE: {results["valid"]["mse"]:.3f}\n'
                      f'Test MSE: {results["test"]["mse"]:.3f}\n'
                      f'Spearman ρ: {metrics["on"]["rho"]:.3f}\n'
                      f'Pearson r: {pearson_on:.3f}')
    ax1.text(0.05, 0.95, metrics_text_on,
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    ax1.set_xlabel('1/Experimental Kd')
    ax1.set_ylabel('Predicted Enrichment')
    ax1.set_title('On-DNA Predictions')
    
    # Off-DNA plot
    ax2.scatter(y_true_off_inv, preds['off']['pred'], alpha=0.5)
    metrics_text_off = (f'Spearman ρ: {metrics["off"]["rho"]:.3f}\n'
                       f'Pearson r: {pearson_off:.3f}')
    ax2.text(0.05, 0.95, metrics_text_off,
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    ax2.set_xlabel('1/Experimental Kd')
    ax2.set_ylabel('Predicted Enrichment')
    ax2.set_title('Off-DNA Predictions')
    
    plt.suptitle(f"ddr1_predictions ({dataset_type})")
    
    # Save plot
    output_file = os.path.join(output_dir, f"ddr1_predictions_{dataset_type}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Save data to CSVs
    for condition in ['on', 'off']:
        df = pd.DataFrame({
            'experimental_kd_inverse': 1/np.log10(preds[condition]['true']),
            'predicted_enrichment': preds[condition]['pred'],
        })
        csv_file = os.path.join(output_dir, f"ddr1_predictions_{dataset_type}_{condition}_DNA.csv")
        df.to_csv(csv_file, index=False)
