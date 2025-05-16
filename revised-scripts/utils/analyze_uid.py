"""
analyze_uid.py - Function for UID metric computations

UID_variance(text): calculate UID variance metric from surprisal values
UID_pairwise(text): same as above, but across pairs of words
vocab_size(text): unique tokens in text
sentence_length(text): number of tokens in text
"""
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
    
    print(f"Loading {len(csv_files)} surprisal files from {directory}...")
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
            vcb_size = vocab_size(df)
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
                'vocab_size': vcb_size,
                'sentence_length': sent_length
            })
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
    
    return pd.DataFrame(metrics)