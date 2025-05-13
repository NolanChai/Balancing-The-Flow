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
import sys
matplotlib.use('Agg')

def load_metrics_files(directories):
    all_metrics = []
    
    for directory in directories:
        dir_path = Path(directory)
        for dataset_dir in dir_path.iterdir():
            if not dataset_dir.is_dir(): # skip non-dir files
                continue
            metrics_file = dataset_dir / "uid_metrics.csv"
            
            if metrics_file.exists():
                try:
                    model_name = dir_path.name
                    dataset_name = dataset_dir.name
                    df = pd.read_csv(metrics_file)
                    df['model'] = model_name
                    df['dataset'] = dataset_name
                    all_metrics.append(df)
                    print(f"Loaded metrics from {model_name}/{dataset_name}: {len(df)} texts")
                except Exception as e:
                    print(f"Error loading {metrics_file}: {e}")
            else:
                print(f"No metrics file found in {directory}")
        print(f"Total files loaded: {len(all_metrics)}")
    
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
    
    # x-axis limits for each metric
    x_limits = {
        'mean_surprisal': None,
        'uid_variance': (0, 100),
        'uid_pairwise': (0, 100)
    }
    
    for metric in metrics_to_compare:
        plt.figure(figsize=(12, 6))
        
        for model in models:
            model_data = metrics_df[metrics_df['model'] == model]
            
            if metric in ['uid_variance', 'uid_pairwise']:
                plot_data = model_data[model_data[metric] <= 100][metric].dropna()
            else:
                plot_data = model_data[metric].dropna()
                
            sns.kdeplot(
                plot_data,
                label=model,
                color=color_dict[model],
                fill=True,
                alpha=0.0
            )
        
        plt.title(f'Distribution of {metric_labels[metric]} Across Sources')
        plt.xlabel(metric_labels[metric])
        plt.ylabel('Density')
        plt.legend(title='Source')
        plt.grid(True, alpha=0.3)
        
        if x_limits[metric]:
            plt.xlim(x_limits[metric])
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}_density.png'), dpi=300)
        plt.close()
    
    for metric in metrics_to_compare:
        plt.figure(figsize=(10, 6))
        
        if metric in ['uid_variance', 'uid_pairwise']:
            plot_df = metrics_df[metrics_df[metric] <= 100].copy()
        else:
            plot_df = metrics_df.copy()
        
        sns.boxplot(
            x='model', 
            y=metric, 
            data=plot_df,
            hue='model',  
            width=0.6
        )
        
        sns.stripplot(
            x='model', 
            y=metric, 
            data=plot_df,
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
        
        # set axis limits if this is uid_variance or uid_pairwise
        if metric in ['uid_variance', 'uid_pairwise']:
            plt.ylim(0, 100)
            
        # remove duplicate legend
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
        
        # create scatterplot with regression lines for each model
        for i, model in enumerate(models):
            try:
                model_data = metrics_df[metrics_df['model'] == model].copy()
                if m1 in ['uid_variance', 'uid_pairwise']:
                    model_data = model_data[model_data[m1] <= 100]
                if m2 in ['uid_variance', 'uid_pairwise']:
                    model_data = model_data[model_data[m2] <= 100]
                
                model_data = model_data.dropna(subset=[m1, m2])
                
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
        
        if m1 in ['uid_variance', 'uid_pairwise']:
            plt.xlim(0, 100)
        if m2 in ['uid_variance', 'uid_pairwise']:
            plt.ylim(0, 100)
            
        plt.legend(title='Source')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{m1}_{m2}_scatter.png'), dpi=300)
        plt.close()
    
    for metric in metrics_to_compare:
        plt.figure(figsize=(12, 7))
        
        if metric in ['uid_variance', 'uid_pairwise']:
            plot_df = metrics_df[metrics_df[metric] <= 100].copy()
        else:
            plot_df = metrics_df.copy()
        
        sns.violinplot(
            x='model',
            y=metric,
            data=plot_df,
            hue='model',
            inner='quartile',
            cut=0
        )
        
        plt.title(f'Distribution of {metric_labels[metric]} by Source')
        plt.xlabel('Source')
        plt.ylabel(metric_labels[metric])
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        if metric in ['uid_variance', 'uid_pairwise']:
            plt.ylim(0, 100)
            
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
        
        print(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during comparison analysis: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())