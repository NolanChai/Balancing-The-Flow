import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import matplotlib
import sys
matplotlib.use('Agg')

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def UID_variance(text):
    """Calculate UID variance metric from surprisal values"""
    N = text.shape[0]
    if N == 0:
        return np.nan
    mu = text['surprisal'].mean()
    surprisals = text['surprisal']
    return ((surprisals - mu) ** 2).sum() / N

def UID_pairwise(text):
    """Calculate UID pairwise metric from surprisal values"""
    N = text.shape[0]
    if N <= 1:
        return np.nan
    surprisals = text['surprisal']
    return (surprisals.diff() ** 2).dropna().sum() / (N - 1)

def vocab_size(text):
    return text['token'].nunique()

def sentence_length(text):
    fulltext = "".join(text['token'])
    sentences = sent_tokenize(fulltext)
    N = len(sentences)
    if N == 0:
        return np.nan
    return sum(len(s.split()) for s in sentences) / N

def load_surprisal_files(directory, regenerate=False):
    """Load all CSV files from a directory into a list of dataframes"""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")
    
    data = []
    filenames = []
    
    print(f"Loading {len(csv_files)} surprisal files...")
    for file in tqdm(csv_files):
        try:
            df = pd.read_csv(file)
            if 'surprisal' in df.columns:
                data.append(df)
                filenames.append(os.path.basename(file))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return data, filenames

def analyze_uid_metrics(data, filenames, regenerate=False):
    """Calculate UID metrics for each text"""
    metrics = []
    
    print("Calculating UID metrics...")
    for i, (df, filename) in enumerate(tqdm(zip(data, filenames))):
        try:
            tokens = len(df)
            mean_surprisal = df['surprisal'].mean()
            median_surprisal = df['surprisal'].median()
            max_surprisal = df['surprisal'].max()
            min_surprisal = df['surprisal'].min()
            
            uid_var = UID_variance(df)
            uid_pair = UID_pairwise(df)
            vocab_size = vocab_size(df)
            sent_length = sentence_length(df)
            
            metrics.append({
                'filename': filename,
                'tokens': tokens,
                'mean_surprisal': mean_surprisal,
                'median_surprisal': median_surprisal,
                'min_surprisal': min_surprisal,
                'max_surprisal': max_surprisal,
                'uid_variance': uid_var,
                'uid_pairwise': uid_pair,
                'vocab_size': vocab_size,
                'sentence_length': sent_length
            })
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
    
    return pd.DataFrame(metrics)

def plot_distributions(metrics_df, output_dir):
    """Create distribution plots for the UID metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    sns.histplot(metrics_df['mean_surprisal'].dropna(), kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Mean Surprisal')
    axes[0, 0].set_xlabel('Mean Surprisal')
    
    sns.histplot(metrics_df['uid_variance'].dropna(), kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of UID Variance')
    axes[0, 1].set_xlabel('UID Variance')
    
    sns.histplot(metrics_df['uid_pairwise'].dropna(), kde=True, ax=axes[0, 2])
    axes[0, 2].set_title('Distribution of UID Pairwise')
    axes[0, 2].set_xlabel('UID Pairwise')
    
    sns.scatterplot(x='mean_surprisal', y='uid_variance', data=metrics_df, ax=axes[1, 0])
    axes[1, 0].set_title('Mean Surprisal vs UID Variance')
    axes[1, 0].set_xlabel('Mean Surprisal')
    axes[1, 0].set_ylabel('UID Variance')
    
    sns.scatterplot(x='mean_surprisal', y='uid_pairwise', data=metrics_df, ax=axes[1, 1])
    axes[1, 1].set_title('Mean Surprisal vs UID Pairwise')
    axes[1, 1].set_xlabel('Mean Surprisal')
    axes[1, 1].set_ylabel('UID Pairwise')
    
    sns.scatterplot(x='uid_variance', y='uid_pairwise', data=metrics_df, ax=axes[1, 2])
    axes[1, 2].set_title('UID Variance vs UID Pairwise')
    axes[1, 2].set_xlabel('UID Variance')
    axes[1, 2].set_ylabel('UID Pairwise')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uid_distributions.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    metrics_for_box = metrics_df[['uid_variance', 'uid_pairwise']].melt()
    sns.boxplot(x='variable', y='value', data=metrics_for_box)
    plt.title('Box Plot of UID Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uid_boxplot.png'), dpi=300)
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(x='tokens', y='uid_variance', data=metrics_df, ax=axes[0])
    axes[0].set_title('Text Length vs UID Variance')
    axes[0].set_xlabel('Number of Tokens')
    axes[0].set_ylabel('UID Variance')
    
    sns.scatterplot(x='tokens', y='uid_pairwise', data=metrics_df, ax=axes[1])
    axes[1].set_title('Text Length vs UID Pairwise')
    axes[1].set_xlabel('Number of Tokens')
    axes[1].set_ylabel('UID Pairwise')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tokens_vs_uid.png'), dpi=300)
    plt.close()

def plot_surprisal_trends(data, filenames, output_dir, sample_size=5):
    """Plot surprisal trends for a sample of texts"""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(data) <= sample_size:
        sample_indices = range(len(data))
    else:
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
    
    for idx in sample_indices:
        df = data[idx]
        filename = filenames[idx]
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['surprisal'].values)
        plt.title(f'Surprisal Trend for {filename}')
        plt.xlabel('Token Position')
        plt.ylabel('Surprisal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trend_{filename}.png'), dpi=300)
        plt.close()
    
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(sample_indices):
        df = data[idx]
        surprisals = df['surprisal'].values
        x_norm = np.linspace(0, 100, len(surprisals))
        plt.plot(x_norm, surprisals, label=f'Text {i+1}')
    
    plt.title('Surprisal Trends Comparison')
    plt.xlabel('Position in Text (%)')
    plt.ylabel('Surprisal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_comparison.png'), dpi=300)
    plt.close()

# this analysis section is generated
def generate_report(metrics_df, output_dir):
    """Generate a summary report with key findings"""
    report_path = os.path.join(output_dir, 'uid_analysis_report.txt')
    
    total_files = len(metrics_df)
    total_tokens = metrics_df['tokens'].sum()
    mean_surprisal = metrics_df['mean_surprisal'].mean()
    
    uid_var_mean = metrics_df['uid_variance'].mean()
    uid_var_median = metrics_df['uid_variance'].median()
    uid_var_std = metrics_df['uid_variance'].std()
    
    uid_pair_mean = metrics_df['uid_pairwise'].mean()
    uid_pair_median = metrics_df['uid_pairwise'].median()
    uid_pair_std = metrics_df['uid_pairwise'].std()
    
    corr_var_pair = metrics_df['uid_variance'].corr(metrics_df['uid_pairwise'])
    corr_mean_var = metrics_df['mean_surprisal'].corr(metrics_df['uid_variance'])
    corr_mean_pair = metrics_df['mean_surprisal'].corr(metrics_df['uid_pairwise'])
    
    with open(report_path, 'w') as f:
        f.write("=== Uniform Information Density (UID) Analysis Report ===\n\n")
        
        f.write(f"Total files analyzed: {total_files}\n")
        f.write(f"Total tokens processed: {total_tokens}\n")
        f.write(f"Average tokens per file: {total_tokens / total_files:.2f}\n\n")
        
        f.write("=== Surprisal Statistics ===\n")
        f.write(f"Mean surprisal across all texts: {mean_surprisal:.4f}\n\n")
        
        f.write("=== UID Metrics ===\n")
        f.write(f"UID Variance (mean): {uid_var_mean:.4f}\n")
        f.write(f"UID Variance (median): {uid_var_median:.4f}\n")
        f.write(f"UID Variance (std): {uid_var_std:.4f}\n\n")
        
        f.write(f"UID Pairwise (mean): {uid_pair_mean:.4f}\n")
        f.write(f"UID Pairwise (median): {uid_pair_median:.4f}\n")
        f.write(f"UID Pairwise (std): {uid_pair_std:.4f}\n\n")
        
        f.write("=== Correlations ===\n")
        f.write(f"Correlation between UID Variance and UID Pairwise: {corr_var_pair:.4f}\n")
        f.write(f"Correlation between Mean Surprisal and UID Variance: {corr_mean_var:.4f}\n")
        f.write(f"Correlation between Mean Surprisal and UID Pairwise: {corr_mean_pair:.4f}\n\n")
        
        f.write("=== Top 5 Files by UID Variance (most uniform) ===\n")
        top_uid_var = metrics_df.sort_values('uid_variance').head(5)
        for i, row in top_uid_var.iterrows():
            f.write(f"{row['filename']}: {row['uid_variance']:.4f}\n")
        
        f.write("\n=== Top 5 Files by UID Pairwise (most uniform) ===\n")
        top_uid_pair = metrics_df.sort_values('uid_pairwise').head(5)
        for i, row in top_uid_pair.iterrows():
            f.write(f"{row['filename']}: {row['uid_pairwise']:.4f}\n")
        
        f.write("\n=== Analysis Notes ===\n")
        f.write("Lower values of UID metrics indicate more uniform information density.\n")
        f.write("Higher correlation between metrics suggests consistency in measuring uniformity.\n")
        
        if corr_var_pair > 0.7:
            f.write("\nThe high correlation between UID Variance and Pairwise suggests both metrics\n")
            f.write("are capturing similar aspects of information uniformity.\n")
        
        if uid_var_std / uid_var_mean > 0.5:
            f.write("\nThe high variability in UID metrics across texts suggests significant\n")
            f.write("differences in information density patterns between texts.\n")
    
    print(f"Report generated at {report_path}")
    
    csv_path = os.path.join(output_dir, 'uid_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Full metrics saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze surprisal data and calculate UID metrics')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing surprisal CSV files')
    parser.add_argument('--output-dir', type=str, default='../UID_Analysis', help='Directory for output files')
    parser.add_argument('--sample-size', type=int, default=5, help='Number of texts to sample for trend plots')
    parser.add_argument('-r', '--regenerate', action="store_true", help='regenerate existing UID analyses')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    if args.output_dir == "infer":
        output_dir = Path('../UID_Analysis') / input_dir.name
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_dir in input_dir.iterdir():
        try:
            if not dataset_dir.is_dir():
                print(f"Skipping {dataset_dir}: Not a directory")
                continue
            dataset_output_dir = output_dir / dataset_dir.name
            data, filenames = load_surprisal_files(dataset_dir, regenerate=args.regenerate)
            print(f"Loaded {len(data)} valid surprisal files")
            
            metrics_df = analyze_uid_metrics(data, filenames, regenerate=args.regenerate)
            
            plot_dir = dataset_output_dir / 'plots'
            plot_distributions(metrics_df, plot_dir)
            
            trends_dir = dataset_output_dir / 'trends'
            plot_surprisal_trends(data, filenames, trends_dir, args.sample_size)
            
            generate_report(metrics_df, dataset_output_dir)
            
            print(f"Analysis complete for {dataset_dir}. Results saved to {dataset_output_dir}")
            
        except Exception as e:
            print(f"Error during analysis of {dataset_dir}: {e}")
            continue

if __name__ == "__main__":
    exit(main())