import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from tqdm.auto import tqdm
import matplotlib
import traceback
matplotlib.use('Agg')

def load_metrics_files(directories):
    all_metrics = []
    
    for directory in directories:
        dir_path = Path(directory)
        metrics_file = dir_path / "uid_metrics.csv"
        
        if metrics_file.exists():
            try:
                model_name = dir_path.name
                df = pd.read_csv(metrics_file)
                df['model'] = model_name
                all_metrics.append(df)
                print(f"Loaded metrics from {model_name}: {len(df)} texts")
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")
        else:
            print(f"No metrics file found in {directory}")
    
    if not all_metrics:
        raise ValueError("No valid metrics files found in the provided directories")
    
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    return combined_metrics

def plot_distributions_comparison(metrics_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    models = metrics_df['model'].unique()
    print(f"Creating comparison plots for {len(models)} models/sources")
    
    colors = sns.color_palette("tab10", n_colors=len(models))
    color_dict = dict(zip(models, colors))
    
    metrics_to_compare = [
        'mean_surprisal', 
        'uid_variance', 
        'uid_pairwise'
    ]
    
    metric_labels = {
        'mean_surprisal': 'Mean Surprisal',
        'uid_variance': 'UID Variance',
        'uid_pairwise': 'UID Pairwise'
    }
    
    for metric in metrics_to_compare:
        plt.figure(figsize=(12, 6))
        
        for model in models:
            model_data = metrics_df[metrics_df['model'] == model]
            sns.kdeplot(
                model_data[metric].dropna(),
                label=model,
                color=color_dict[model],
                fill=True,
                alpha=0.3
            )
        
        plt.title(f'Distribution of {metric_labels[metric]} Across Sources')
        plt.xlabel(metric_labels[metric])
        plt.ylabel('Density')
        plt.legend(title='Source')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}_density.png'), dpi=300)
        plt.close()
    
    for metric in metrics_to_compare:
        plt.figure(figsize=(10, 6))
        
        sns.boxplot(
            x='model', 
            y=metric, 
            data=metrics_df,
            hue='model',
            width=0.6
        )
        
        sns.stripplot(
            x='model', 
            y=metric, 
            data=metrics_df,
            size=3, 
            color='black',
            alpha=0.3,
            jitter=True
        )
        
        plt.title(f'{metric_labels[metric]} by Source')
        plt.xlabel('Source')
        plt.ylabel(metric_labels[metric])
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend([],[], frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}_boxplot.png'), dpi=300)
        plt.close()
    
    metric_pairs = [
        ('mean_surprisal', 'uid_variance'),
        ('mean_surprisal', 'uid_pairwise'),
        ('uid_variance', 'uid_pairwise')
    ]
    
    for m1, m2 in metric_pairs:
        plt.figure(figsize=(10, 8))
        
        for i, model in enumerate(models):
            try:
                model_data = metrics_df[metrics_df['model'] == model].dropna(subset=[m1, m2])
                
                plt.scatter(
                    model_data[m1],
                    model_data[m2],
                    label=model,
                    color=color_dict[model],
                    alpha=0.5,
                    s=30
                )
                
                if len(model_data) > 1:
                    try:
                        z = np.polyfit(model_data[m1], model_data[m2], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(model_data[m1].min(), model_data[m1].max(), 100)
                        plt.plot(x_range, p(x_range), color=color_dict[model], linestyle='--', linewidth=2)
                    except Exception as e:
                        print(f"Error creating trend line for {model}: {e}")
            except Exception as e:
                print(f"Error plotting data for {model}: {e}")
        
        plt.title(f'Relationship Between {metric_labels[m1]} and {metric_labels[m2]}')
        plt.xlabel(metric_labels[m1])
        plt.ylabel(metric_labels[m2])
        plt.legend(title='Source')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{m1}_{m2}_scatter.png'), dpi=300)
        plt.close()
    
    for metric in metrics_to_compare:
        plt.figure(figsize=(12, 7))
        
        sns.violinplot(
            x='model',
            y=metric,
            data=metrics_df,
            hue='model',
            inner='quartile',
            cut=0
        )
        
        plt.title(f'Distribution of {metric_labels[metric]} by Source')
        plt.xlabel('Source')
        plt.ylabel(metric_labels[metric])
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend([],[], frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}_violin.png'), dpi=300)
        plt.close()

def generate_summary_statistics(metrics_df, output_dir):
    models = metrics_df['model'].unique()
    
    metrics_to_compare = [
        'tokens', 
        'mean_surprisal', 
        'uid_variance', 
        'uid_pairwise'
    ]
    
    summary_rows = []
    
    for model in models:
        model_data = metrics_df[metrics_df['model'] == model]
        
        row = {'model': model}
        
        for metric in metrics_to_compare:
            row[f'{metric}_mean'] = model_data[metric].mean()
            row[f'{metric}_median'] = model_data[metric].median()
            row[f'{metric}_std'] = model_data[metric].std()
            row[f'{metric}_min'] = model_data[metric].min()
            row[f'{metric}_max'] = model_data[metric].max()
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    summary_file = os.path.join(output_dir, 'uid_comparison_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to {summary_file}")
    
    report_file = os.path.join(output_dir, 'uid_comparison_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=== UID Metrics Comparison Report ===\n\n")
        
        for model in models:
            model_data = metrics_df[metrics_df['model'] == model]
            
            f.write(f"== Source: {model} ==\n")
            f.write(f"Number of texts: {len(model_data)}\n")
            
            for metric in metrics_to_compare:
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean:   {model_data[metric].mean():.4f}\n")
                f.write(f"  Median: {model_data[metric].median():.4f}\n")
                f.write(f"  StdDev: {model_data[metric].std():.4f}\n")
                f.write(f"  Min:    {model_data[metric].min():.4f}\n")
                f.write(f"  Max:    {model_data[metric].max():.4f}\n")
            
            f.write("\n")
        
        f.write("\n=== Statistical Comparison ===\n\n")
        
        if len(models) >= 2:
            try:
                from scipy import stats
                
                for metric in metrics_to_compare:
                    f.write(f"\n== {metric} Comparisons ==\n")
                    
                    for i, model1 in enumerate(models):
                        for model2 in models[i+1:]:
                            try:
                                data1 = metrics_df[metrics_df['model'] == model1][metric].dropna()
                                data2 = metrics_df[metrics_df['model'] == model2][metric].dropna()
                                
                                if len(data1) == 0 or len(data2) == 0:
                                    f.write(f"{model1} vs {model2}: Insufficient data for comparison\n")
                                    continue
                                
                                u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                f.write(f"{model1} vs {model2}:\n")
                                f.write(f"  Mann-Whitney U test: p-value = {p_value:.6f}")
                                
                                if p_value < 0.001:
                                    f.write(" (highly significant difference)\n")
                                elif p_value < 0.01:
                                    f.write(" (very significant difference)\n")
                                elif p_value < 0.05:
                                    f.write(" (significant difference)\n")
                                else:
                                    f.write(" (no significant difference)\n")
                                    
                                mean1, mean2 = data1.mean(), data2.mean()
                                std1, std2 = data1.std(), data2.std()
                                
                                n1, n2 = len(data1), len(data2)
                                if n1 > 1 and n2 > 1 and std1 > 0 and std2 > 0:
                                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                                    
                                    if pooled_std > 0:
                                        cohens_d = abs(mean1 - mean2) / pooled_std
                                        f.write(f"  Effect size (Cohen's d): {cohens_d:.4f}")
                                        
                                        if cohens_d < 0.2:
                                            f.write(" (negligible effect)\n")
                                        elif cohens_d < 0.5:
                                            f.write(" (small effect)\n")
                                        elif cohens_d < 0.8:
                                            f.write(" (medium effect)\n")
                                        else:
                                            f.write(" (large effect)\n")
                                    else:
                                        f.write("  Could not calculate effect size (zero standard deviation)\n")
                                else:
                                    f.write("  Could not calculate effect size (insufficient data or zero standard deviation)\n")
                            
                            except Exception as e:
                                f.write(f"  Error comparing {model1} vs {model2}: {str(e)}\n")
            except Exception as e:
                f.write(f"Error performing statistical comparison: {str(e)}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
    
    print(f"Detailed comparison report saved to {report_file}")
    
    return summary_df

def create_correlation_heatmap(metrics_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    models = metrics_df['model'].unique()
    
    metrics_to_compare = [
        'tokens', 
        'mean_surprisal',
        'median_surprisal',
        'min_surprisal',
        'max_surprisal',
        'uid_variance', 
        'uid_pairwise'
    ]
    
    available_metrics = [m for m in metrics_to_compare if m in metrics_df.columns]
    
    if len(available_metrics) < 2:
        print("Not enough metrics available for correlation analysis")
        return
    
    try:
        plt.figure(figsize=(10, 8))
        
        corr_data = metrics_df[available_metrics].corr()
        
        sns.heatmap(
            corr_data, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            center=0,
            fmt='.2f',
            linewidths=0.5
        )
        
        plt.title('Correlation Between UID Metrics (All Sources)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap_all.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating overall correlation heatmap: {e}")
    
    for model in models:
        try:
            plt.figure(figsize=(10, 8))
            
            model_data = metrics_df[metrics_df['model'] == model][available_metrics]
            if len(model_data) < 2:
                print(f"Not enough data for correlation analysis for model {model}")
                continue
                
            corr_data = model_data.corr()
            
            sns.heatmap(
                corr_data, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                center=0,
                fmt='.2f',
                linewidths=0.5
            )
            
            plt.title(f'Correlation Between UID Metrics ({model})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'correlation_heatmap_{model}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating correlation heatmap for {model}: {e}")

def plot_token_length_vs_metrics(metrics_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    models = metrics_df['model'].unique()
    
    colors = sns.color_palette("tab10", n_colors=len(models))
    color_dict = dict(zip(models, colors))
    
    metrics_to_plot = [
        'mean_surprisal', 
        'uid_variance', 
        'uid_pairwise'
    ]
    
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    if 'tokens' not in metrics_df.columns or len(available_metrics) == 0:
        print("Required columns missing for token length analysis")
        return
    
    metric_labels = {
        'mean_surprisal': 'Mean Surprisal',
        'uid_variance': 'UID Variance',
        'uid_pairwise': 'UID Pairwise'
    }
    
    for metric in available_metrics:
        try:
            plt.figure(figsize=(12, 8))
            
            for model in models:
                model_data = metrics_df[metrics_df['model'] == model].dropna(subset=['tokens', metric])
                if len(model_data) < 2:
                    print(f"Not enough data for scatter plot for model {model}, metric {metric}")
                    continue
                
                plt.scatter(
                    model_data['tokens'],
                    model_data[metric],
                    label=model,
                    color=color_dict[model],
                    alpha=0.5,
                    edgecolor='none'
                )
                
                try:
                    if len(model_data) > 1:
                        z = np.polyfit(model_data['tokens'], model_data[metric], 1)
                        p = np.poly1d(z)
                        x_vals = np.linspace(model_data['tokens'].min(), model_data['tokens'].max(), 100)
                        plt.plot(
                            x_vals,
                            p(x_vals),
                            color=color_dict[model],
                            linestyle='--',
                            linewidth=2
                        )
                except Exception as e:
                    print(f"Error creating trend line for {model}, {metric}: {e}")
            
            plt.title(f'Text Length vs. {metric_labels[metric]}')
            plt.xlabel('Number of Tokens')
            plt.ylabel(metric_labels[metric])
            plt.legend(title='Source')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'tokens_vs_{metric}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating scatter plot for {metric}: {e}")
    
    try:
        if len(metrics_df) > 0:
            min_tokens = max(1, metrics_df['tokens'].min())
            max_tokens = metrics_df['tokens'].max()
            
            if max_tokens <= 500:
                bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', 
                         '300-350', '350-400', '400-450', '450-500']
            else:
                bins = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, float('inf')]
                labels = ['0-100', '100-200', '200-300', '300-400', '400-500', 
                         '500-750', '750-1000', '1000-1500', '1500+']
            
            metrics_df['token_bin'] = pd.cut(
                metrics_df['tokens'],
                bins=bins,
                labels=labels
            )
            
            for metric in available_metrics:
                try:
                    plt.figure(figsize=(14, 8))
                    
                    binned_data = metrics_df.groupby(['model', 'token_bin'])[metric].mean().reset_index()
                    
                    if len(binned_data) > 0:
                        ax = sns.barplot(
                            x='token_bin',
                            y=metric,
                            hue='model',
                            data=binned_data
                        )
                        
                        plt.title(f'Text Length vs. {metric_labels[metric]} (Binned)')
                        plt.xlabel('Token Count Bin')
                        plt.ylabel(f'Average {metric_labels[metric]}')
                        plt.legend(title='Source')
                        plt.grid(True, alpha=0.3, axis='y')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'binned_tokens_vs_{metric}.png'), dpi=300)
                        plt.close()
                    else:
                        print(f"No data available for binned analysis of {metric}")
                except Exception as e:
                    print(f"Error creating binned plot for {metric}: {e}")
    except Exception as e:
        print(f"Error in binned analysis: {e}")

def add_distribution_tests(metrics_df, output_dir):
    """
    Add Kolmogorov-Smirnov tests to compare distributions between models.
    
    Args:
        metrics_df: DataFrame containing metrics for all models
        output_dir: Directory to save the results
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique models
    models = metrics_df['model'].unique()
    
    # Define metrics to compare
    metrics_to_compare = [
        'mean_surprisal', 
        'uid_variance', 
        'uid_pairwise'
    ]
    
    # Skip if there's only one model
    if len(models) < 2:
        print("Need at least two models for distribution comparison tests")
        return
    
    # Create a text file for the results
    results_file = os.path.join(output_dir, 'distribution_test_results.txt')
    
    # Create a dataframe to store all results for plotting
    test_results = []
    
    with open(results_file, 'w') as f:
        f.write("=== Distribution Comparison Tests ===\n\n")
        f.write("Kolmogorov-Smirnov tests compare if two distributions are different.\n")
        f.write("Null hypothesis: The two samples come from the same distribution.\n")
        f.write("Small p-values indicate the distributions are significantly different.\n\n")
        
        # For each metric, compare each pair of models
        for metric in metrics_to_compare:
            f.write(f"\n== {metric} Distribution Tests ==\n\n")
            
            # Create a plot for this metric
            plt.figure(figsize=(10, 6))
            
            # Create a grid for pairwise comparisons
            comparison_grid = np.zeros((len(models), len(models)))
            p_values = np.ones((len(models), len(models)))
            
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i >= j:  # Skip diagonal and duplicates
                        continue
                        
                    try:
                        data1 = metrics_df[metrics_df['model'] == model1][metric].dropna()
                        data2 = metrics_df[metrics_df['model'] == model2][metric].dropna()
                        
                        if len(data1) == 0 or len(data2) == 0:
                            f.write(f"{model1} vs {model2}: Insufficient data for comparison\n")
                            continue
                        
                        # Perform Kolmogorov-Smirnov test
                        ks_stat, p_value = stats.ks_2samp(data1, data2)
                        
                        # Store test statistic in grid
                        comparison_grid[i, j] = ks_stat
                        comparison_grid[j, i] = ks_stat  # Symmetric
                        
                        # Store p-value in grid
                        p_values[i, j] = p_value
                        p_values[j, i] = p_value  # Symmetric
                        
                        # Write results to file
                        f.write(f"{model1} vs {model2}:\n")
                        f.write(f"  KS statistic: {ks_stat:.4f}\n")
                        f.write(f"  p-value: {p_value:.8f}")
                        
                        # Add significance markers
                        if p_value < 0.001:
                            f.write(" (*** highly significant difference)\n")
                        elif p_value < 0.01:
                            f.write(" (** very significant difference)\n")
                        elif p_value < 0.05:
                            f.write(" (* significant difference)\n")
                        else:
                            f.write(" (not significant)\n")
                        
                        # Add info on which distribution has higher values
                        mean1 = data1.mean()
                        mean2 = data2.mean()
                        median1 = data1.median()
                        median2 = data2.median()
                        
                        f.write(f"  {model1}: mean={mean1:.4f}, median={median1:.4f}\n")
                        f.write(f"  {model2}: mean={mean2:.4f}, median={median2:.4f}\n")
                        
                        if mean1 > mean2:
                            f.write(f"  {model1} has higher average values than {model2}\n")
                        else:
                            f.write(f"  {model2} has higher average values than {model1}\n")
                        f.write("\n")
                        
                        # Store results for plotting
                        test_results.append({
                            'metric': metric,
                            'model1': model1,
                            'model2': model2,
                            'ks_statistic': ks_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })
                        
                    except Exception as e:
                        f.write(f"  Error comparing {model1} vs {model2}: {str(e)}\n\n")
            
            # Plot heatmap of test statistics
            plt.figure(figsize=(10, 8))
            mask = np.zeros_like(comparison_grid, dtype=bool)
            mask[np.tril_indices_from(mask)] = True  # Mask lower triangle including diagonal
            
            sns.heatmap(
                comparison_grid, 
                annot=True, 
                cmap='viridis', 
                xticklabels=models,
                yticklabels=models,
                mask=mask,
                fmt='.3f',
                vmin=0
            )
            
            plt.title(f'Kolmogorov-Smirnov Test Statistic for {metric}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ks_test_{metric}.png'), dpi=300)
            plt.close()
            
            # Plot heatmap of p-values with significance highlighting
            plt.figure(figsize=(10, 8))
            
            # Log transform p-values for better visualization (and handle zeros)
            log_p = -np.log10(np.maximum(p_values, 1e-10))  # -log10(p), with minimum p of 1e-10
            
            sns.heatmap(
                log_p, 
                annot=p_values,  # Show actual p-values
                cmap='YlOrRd', 
                xticklabels=models,
                yticklabels=models,
                mask=mask,
                fmt='.3g',  # Scientific notation for small values
                vmin=0
            )
            
            plt.title(f'p-values for {metric} (Kolmogorov-Smirnov Test)\n-log10(p) color scale: darker = more significant')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ks_pvalue_{metric}.png'), dpi=300)
            plt.close()
        
        # Summary section
        f.write("\n\n=== Summary of Significant Differences ===\n\n")
        for metric in metrics_to_compare:
            sig_tests = [t for t in test_results if t['metric'] == metric and t['significant']]
            if sig_tests:
                f.write(f"{metric}: {len(sig_tests)} significant differences found\n")
                for test in sig_tests:
                    f.write(f"  {test['model1']} vs {test['model2']}: p={test['p_value']:.6f}\n")
            else:
                f.write(f"{metric}: No significant differences found\n")
            f.write("\n")
    
    print(f"Distribution tests completed and saved to {results_file}")
    
    # Create distribution comparison plots with overlay of KS test p-values
    for metric in metrics_to_compare:
        plt.figure(figsize=(12, 8))
        
        # Create a distplot for each model
        metric_data = []
        for model in models:
            data = metrics_df[metrics_df['model'] == model][metric].dropna()
            sns.kdeplot(data, label=f"{model}", fill=True, alpha=0.3)
            metric_data.append((model, data))
        
        # Add KS test p-values as annotations
        for i in range(len(metric_data)):
            for j in range(i+1, len(metric_data)):
                model1, data1 = metric_data[i]
                model2, data2 = metric_data[j]
                
                try:
                    _, p_value = stats.ks_2samp(data1, data2)
                    
                    # Determine y position for annotation
                    y_pos = 0.95 - 0.05 * (i * len(metric_data) + j)
                    
                    # Add significance markers
                    sig_markers = ""
                    if p_value < 0.001:
                        sig_markers = "***"
                    elif p_value < 0.01:
                        sig_markers = "**"
                    elif p_value < 0.05:
                        sig_markers = "*"
                    
                    plt.annotate(
                        f"{model1} vs {model2}: p={p_value:.4g} {sig_markers}",
                        xy=(0.05, y_pos),
                        xycoords='axes fraction',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                    )
                except:
                    pass
        
        plt.title(f'Distribution of {metric} with KS Test Results')
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.legend(title='Source')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribution_{metric}_with_ks.png'), dpi=300)
        plt.close()
    
    return results_file

def main():
    parser = argparse.ArgumentParser(description='Compare UID metrics across different models and human texts')
    parser.add_argument('--directories', type=str, nargs='+', required=True, 
                        help='Directories containing UID analysis results to compare')
    parser.add_argument('--output-dir', type=str, default='../UID_Comparison', 
                        help='Directory for output files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        metrics_df = load_metrics_files(args.directories)
        print(f"Loaded {len(metrics_df)} total text metrics from {len(args.directories)} sources")
        
        plot_distributions_comparison(metrics_df, plots_dir)
        
        summary_df = generate_summary_statistics(metrics_df, args.output_dir)
        
        create_correlation_heatmap(metrics_df, plots_dir)
        
        plot_token_length_vs_metrics(metrics_df, plots_dir)
        
        if len(metrics_df['model'].unique()) >= 2:
            print("Performing distribution comparison tests...")
            results_file = add_distribution_tests(metrics_df, plots_dir)
            print(f"Distribution test results saved to {results_file}")
        
        print(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during comparison analysis: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())