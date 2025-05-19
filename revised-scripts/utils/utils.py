import os
import re
import sys
import json
import datetime
import pandas as pd
from pathlib import Path
from utils.spinner import Spinner
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import math
import matplotlib.pyplot as plt
import tqdm

from datasets import load_dataset, Dataset # type: ignore

"""
Timestamped logging
"""
def log_time():
    now = datetime.datetime.now()
    log = f"[{now:%Y-%m-%d %H:%M:%S}]"
    return log
def log(message):
    print(f"{log_time()} "+message)

def get_dataset(dataset_name, split='test', cache_dir="datasets_cache", seed=42):
    dataset_config = ""
    if "/" in dataset_name:
        parts = dataset_name.split("/")
        if len(parts) > 2:
            dataset_name_parsed = "/".join(parts[:2])
            dataset_config = "/".join(parts[2:])
        else:
            dataset_name_parsed = dataset_name
    else:
        dataset_name_parsed = dataset_name
    
    dataset_id = dataset_name.replace("/", "_")
    cache_path = Path(cache_dir) / f"{dataset_id}_{split}"
    cache_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = cache_path / "metadata.json"
    data_path = cache_path / "data.json"
    
    if metadata_path.exists() and data_path.exists():
        with Spinner(message=f"{log_time()} Loading cached dataset {dataset_id}", spinner_type="braille") as spinner:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if (metadata['dataset'] == dataset_name and 
                    metadata['split'] == split):
                    
                    with open(data_path, 'r') as f:
                        data_dict = json.load(f)
                    
                    dataset = Dataset.from_dict(data_dict)
                    shuffled_data = dataset.shuffle(seed=seed)
                    
                    spinner.stop(success=True)
                    log(f"Loaded dataset {dataset_id} from cache with {len(dataset)} examples")
                    return shuffled_data, True
            
            except Exception as e:
                spinner.stop(success=False)
                log(f"Failed to load cached dataset: {str(e)}")
    
    with Spinner(message=f"{log_time()} Downloading dataset {dataset_name}", spinner_type="braille") as spinner:
        try:
            dataset_params = (dataset_name_parsed,)
            if dataset_config:
                dataset_params += (dataset_config,)
            
            if dataset_name == "abisee/cnn_dailymail":
                data = load_dataset(*dataset_params, '1.0.0', trust_remote_code=True, split=split)
            else:
                data = load_dataset(*dataset_params, trust_remote_code=True, split=split)
            
            metadata = {
                'dataset': dataset_name,
                'split': split,
                'size': len(data),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            with open(data_path, 'w') as f:
                json.dump(data.to_dict(), f)
            
            shuffled_data = data.shuffle(seed=seed)
            
            spinner.stop(success=True)
            log(f"Downloaded dataset {dataset_id} with {len(data)} examples and saved to cache")
            return shuffled_data, False
            
        except Exception as e:
            spinner.stop(success=False)
            log(f"ERROR: Failed to download dataset {dataset_name}: {str(e)}")
            raise

def consolidate_generations(generations_dir, output_dir=None, use_rich=True):
    """
    Consolidate all text file generations into dataset-specific CSVs.
    
    Args:
        generations_dir: Path to the main Generations folder
        output_dir: Directory to save the consolidated CSVs (defaults to {generations_dir}/consolidated)
        use_rich: Whether to use rich progress bars if available
    
    Returns:
        Dict mapping dataset names to output CSV paths
    """
    generations_dir = Path(generations_dir)
    
    if not output_dir:
        output_dir = generations_dir / "consolidated"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all model folders
    model_dirs = [d for d in generations_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    with Spinner(message=f"[{log_time()}] Finding all generation files", spinner_type="braille") as spinner:
        # Create a mapping of dataset -> [files]
        dataset_files = {}
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            
            # Skip the consolidated directory if it's there
            if model_name == "consolidated":
                continue
                
            # Find dataset directories for this model
            dataset_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            
            for dataset_dir in dataset_dirs:
                dataset_name = dataset_dir.name
                
                # Initialize dataset entry if it doesn't exist
                if dataset_name not in dataset_files:
                    dataset_files[dataset_name] = []
                
                # Find all text files for this dataset
                txt_files = list(dataset_dir.glob("*.txt"))
                
                # Add each file with metadata
                for txt_file in txt_files:
                    dataset_files[dataset_name].append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'filename': txt_file.name,
                        'filepath': txt_file
                    })
        
        spinner.stop(success=True)
    
    # Process each dataset
    results = {}
    
    try:
        from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
        from rich.progress import TimeRemainingColumn, MofNCompleteColumn
        from rich.console import Console
        RICH_AVAILABLE = True and use_rich
    except ImportError:
        RICH_AVAILABLE = False
    
    if RICH_AVAILABLE:
        console = Console(file=sys.stdout)
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            expand=True,
            console=console
        ) as progress:
            datasets_task = progress.add_task("[cyan]Processing datasets", total=len(dataset_files))
            
            for dataset_name, files in dataset_files.items():
                progress.update(datasets_task, description=f"[cyan]Processing {dataset_name} ({len(files)} files)")
                
                # Create list to hold all data for this dataset
                all_data = []
                
                # Add a nested progress bar for files in this dataset
                file_task = progress.add_task(f"[blue]Reading {dataset_name} files", total=len(files))
                
                for file_info in files:
                    progress.update(file_task, description=f"[blue]Reading {file_info['filename']}")
                    
                    try:
                        with open(file_info['filepath'], 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Extract ID from filename
                        match = re.search(r'_(\d+)\.txt$', file_info['filename'])
                        gen_id = match.group(1) if match else "unknown"
                        
                        all_data.append({
                            'model': file_info['model'],
                            'dataset': dataset_name,
                            'generation_id': gen_id,
                            'filename': file_info['filename'],
                            'content': content
                        })
                    except Exception as e:
                        print(f"Error reading {file_info['filepath']}: {str(e)}")
                    
                    progress.update(file_task, advance=1)
                
                # Convert to DataFrame and save
                if all_data:
                    df = pd.DataFrame(all_data)
                    output_file = output_dir / f"{dataset_name}_consolidated.csv"
                    df.to_csv(output_file, index=False)
                    results[dataset_name] = output_file
                
                progress.update(datasets_task, advance=1)
                # Remove the completed file task
                progress.remove_task(file_task)
    else:
        # Use tqdm for progress tracking
        for dataset_name, files in tqdm(dataset_files.items(), desc="Processing datasets"):
            print(f"Processing {dataset_name} ({len(files)} files)")
            
            # Create list to hold all data for this dataset
            all_data = []
            
            for file_info in tqdm(files, desc=f"Reading {dataset_name} files"):
                try:
                    with open(file_info['filepath'], 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Extract ID from filename
                    match = re.search(r'_(\d+)\.txt$', file_info['filename'])
                    gen_id = match.group(1) if match else "unknown"
                    
                    all_data.append({
                        'model': file_info['model'],
                        'dataset': dataset_name,
                        'generation_id': gen_id,
                        'filename': file_info['filename'],
                        'content': content
                    })
                except Exception as e:
                    print(f"Error reading {file_info['filepath']}: {str(e)}")
            
            # Convert to DataFrame and save
            if all_data:
                df = pd.DataFrame(all_data)
                output_file = output_dir / f"{dataset_name}_consolidated.csv"
                df.to_csv(output_file, index=False)
                results[dataset_name] = output_file
    
    log(f"Consolidated {len(results)} datasets into CSV files in {output_dir}")
    
    return results

def calculate_surprisals(csv_files, tokenizer_source="gpt2", max_length=1024, output_dir=None):
    """
    Calculate token-level surprisals for text in CSV files.
    
    Args:
        csv_files: List of CSV file paths
        tokenizer_source: Either a string for HuggingFace model name (e.g., "gpt2") 
                         or a path to a tokenizer JSON file
        max_length: Maximum sequence length for tokenization
        output_dir: Directory to save output CSV files (if None, results are not saved)
        
    Returns:
        A dictionary mapping file names to dataframes with text and their surprisals
    """
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    if tokenizer_source.endswith('.json'):
        # Load from JSON file
        with open(tokenizer_source, 'r') as f:
            tokenizer_json = json.load(f)
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",  # Use gpt2 as base
                vocab_file=None,
                merges_file=None,
                tokenizer_file=None
            )
            # Update with custom vocabulary
            tokenizer.vocab = tokenizer_json.get('vocab', {})
            tokenizer.ids_to_tokens = {v: k for k, v in tokenizer.vocab.items()}
    else:
        # Load from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    
    # Load a matching model based on tokenizer
    model_name = "gpt2" if tokenizer_source == "gpt2" or tokenizer_source.endswith('.json') else tokenizer_source
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    results = {}
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        
        # Read the CSV
        df = pd.read_csv(csv_file)
        
        # Identify the text column - look for common text column names
        text_col = None
        possible_text_cols = ['text', 'content', 'article', 'dialogue', 'prompt', 'message']
        for col in possible_text_cols:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            # If no matching column found, use the first string column
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
                    
        if text_col is None:
            print(f"No text column found in {csv_file}, skipping.")
            continue
            
        # Create a new dataframe to store results
        result_df = pd.DataFrame()
        result_df['original_text'] = df[text_col]
        
        # Keep all original columns from the source dataframe
        for col in df.columns:
            if col != text_col:  # Skip the text column as we already have it as 'original_text'
                result_df[col] = df[col]
        
        # Store token-level surprisals
        all_surprisals = []
        all_tokens = []
        
        for text in df[text_col]:
            # Skip NaN or empty texts
            if pd.isna(text) or text == '':
                all_surprisals.append([])
                all_tokens.append([])
                continue
                
            # Tokenize
            token_ids = tokenizer.encode(text, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
            
            with torch.no_grad():
                # Get model output
                outputs = model(token_ids, labels=token_ids)
                logits = outputs.logits
                
                # Shift logits and tokens for calculating next token prediction
                shift_logits = logits[0, :-1, :].contiguous()
                shift_tokens = token_ids[0, 1:].contiguous()
                
                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                
                # Get probability of each actual next token
                indices = torch.arange(shift_tokens.size(0))
                token_probs = probs[indices, shift_tokens]
                
                # Calculate surprisal as -log(p)
                # Ensure operation order: first calculate log2, then negate, then convert to list
                log_probs = torch.log2(token_probs)
                surprisals = -log_probs
                token_surprisals = surprisals.tolist()
                
                # Add padding of 0 for the first token (which doesn't have a preceding context)
                token_surprisals = [0.0] + token_surprisals
                
                all_surprisals.append(token_surprisals)
                all_tokens.append(tokens)
        
        # Add surprisals to result dataframe
        result_df['token_surprisals'] = all_surprisals
        result_df['tokens'] = all_tokens
        
        # Calculate average surprisal per text
        result_df['avg_surprisal'] = result_df['token_surprisals'].apply(
            lambda x: np.mean(x) if len(x) > 0 else np.nan
        )
        
        # Create token-level dataframe for detailed analysis
        token_rows = []
        for i, (surprisals, tokens, text) in enumerate(zip(all_surprisals, all_tokens, result_df['original_text'])):
            for j, (token, surprisal) in enumerate(zip(tokens, surprisals)):
                row = {
                    'text_id': i,
                    'token_id': j,
                    'token': token,
                    'surprisal': surprisal,
                    'text': text
                }
                token_rows.append(row)
        
        token_df = pd.DataFrame(token_rows)
        
        # Store both dataframes in the results dictionary
        results[csv_file] = {
            'text_level': result_df,
            'token_level': token_df
        }
        
        # Save to CSV if output_dir is provided
        if output_dir is not None:
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Save text-level dataframe
            # Convert token_surprisals list to string for CSV storage
            result_df_for_csv = result_df.copy()
            result_df_for_csv['token_surprisals'] = result_df_for_csv['token_surprisals'].apply(lambda x: ','.join([str(val) for val in x]) if len(x) > 0 else '')
            result_df_for_csv['tokens'] = result_df_for_csv['tokens'].apply(lambda x: ','.join([str(val) for val in x]) if len(x) > 0 else '')
            
            result_df_for_csv.to_csv(os.path.join(output_dir, f"{base_name}_text_surprisals.csv"), index=False)
            
            # Save token-level dataframe
            token_df.to_csv(os.path.join(output_dir, f"{base_name}_token_surprisals.csv"), index=False)
            
            print(f"Saved results for {csv_file} to {output_dir}")
    
    return results

def process_csv_files_with_surprisals(csv_files, tokenizer_source="gpt2", max_length=1024, output_dir="surprisal_results"):
    """
    Process CSV files and calculate surprisals, saving results to CSV files.
    
    Args:
        csv_files: List of CSV file paths
        tokenizer_source: Either "gpt2" or path to a tokenizer JSON file
        max_length: Maximum sequence length for tokenization
        output_dir: Directory to save output CSV files
    """
    # Calculate surprisals
    results = calculate_surprisals(
        csv_files=csv_files,
        tokenizer_source=tokenizer_source,
        max_length=max_length,
        output_dir=output_dir
    )
    
    # Print summary statistics
    for file, data in results.items():
        if 'text_level' in data and 'avg_surprisal' in data['text_level'].columns:
            text_df = data['text_level']
            token_df = data['token_level']
            
            print(f"\nSummary for {file}:")
            print(f"  - Total texts: {len(text_df)}")
            print(f"  - Total tokens: {len(token_df)}")
            print(f"  - Average surprisal: {text_df['avg_surprisal'].mean():.4f}")
            print(f"  - Min surprisal: {text_df['avg_surprisal'].min():.4f}")
            print(f"  - Max surprisal: {text_df['avg_surprisal'].max():.4f}")
            
    return results

def analyze_token_lengths(csv_files, tokenizer_name="gpt2", threshold=1024):
    """
    Analyze how many texts in CSV files exceed a token length threshold.
    
    Args:
        csv_files: List of CSV file paths
        tokenizer_name: HuggingFace tokenizer to use (default: "gpt2")
        threshold: Token count threshold to check against (default: 1024)
        
    Returns:
        A dictionary with statistics for each file
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    results = {}
    
    for csv_file in csv_files:
        print(f"Analyzing {csv_file}...")
        
        try:
            # Read the CSV
            df = pd.read_csv(csv_file)
            
            # Identify the text column - look for common text column names
            text_col = None
            possible_text_cols = ['text', 'content', 'article', 'dialogue', 'prompt', 'message']
            for col in possible_text_cols:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                # If no matching column found, use the first string column
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_col = col
                        break
                        
            if text_col is None:
                print(f"No text column found in {csv_file}, skipping.")
                continue
                
            # Analyze token lengths
            token_counts = []
            for text in df[text_col]:
                # Skip NaN or empty texts
                if pd.isna(text) or text == '':
                    token_counts.append(0)
                    continue
                    
                # Count tokens
                tokens = tokenizer.encode(text)
                token_counts.append(len(tokens))
            
            # Add token counts to the dataframe for reference
            token_count_col = 'token_count'
            temp_df = df.copy()
            temp_df[token_count_col] = token_counts
            
            # Calculate statistics
            total_texts = len(temp_df)
            exceeding_threshold = sum(temp_df[token_count_col] > threshold)
            percentage = (exceeding_threshold / total_texts) * 100 if total_texts > 0 else 0
            
            # Distribution statistics
            max_tokens = max(token_counts) if token_counts else 0
            min_tokens = min(token_counts) if token_counts else 0
            mean_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            median_tokens = sorted(token_counts)[len(token_counts) // 2] if token_counts else 0
            
            # Calculate percentiles
            percentiles = {
                '90th': np.percentile(token_counts, 90) if token_counts else 0,
                '95th': np.percentile(token_counts, 95) if token_counts else 0,
                '99th': np.percentile(token_counts, 99) if token_counts else 0
            }
            
            # Group by token count ranges
            bins = [0, 256, 512, 768, 1024, 1536, 2048, float('inf')]
            bin_labels = ['0-256', '257-512', '513-768', '769-1024', '1025-1536', '1537-2048', '2048+']
            temp_df['token_range'] = pd.cut(temp_df[token_count_col], bins=bins, labels=bin_labels)
            range_counts = temp_df['token_range'].value_counts().sort_index()
            
            # Store results
            file_results = {
                'file_name': csv_file,
                'total_texts': total_texts,
                'exceeding_threshold': exceeding_threshold,
                'percentage_exceeding': percentage,
                'max_tokens': max_tokens,
                'min_tokens': min_tokens,
                'mean_tokens': mean_tokens,
                'median_tokens': median_tokens,
                'percentiles': percentiles,
                'token_counts': token_counts,
                'range_distribution': range_counts.to_dict()
            }
            
            results[csv_file] = file_results
            
            # Display summary
            print(f"Results for {csv_file}:")
            print(f"  - Total texts: {total_texts}")
            print(f"  - Texts exceeding {threshold} tokens: {exceeding_threshold} ({percentage:.2f}%)")
            print(f"  - Token count statistics:")
            print(f"    - Min: {min_tokens}")
            print(f"    - Mean: {mean_tokens:.2f}")
            print(f"    - Median: {median_tokens}")
            print(f"    - Max: {max_tokens}")
            print(f"    - 90th percentile: {percentiles['90th']:.2f}")
            print(f"    - 95th percentile: {percentiles['95th']:.2f}")
            print(f"    - 99th percentile: {percentiles['99th']:.2f}")
            print(f"  - Distribution by token count range:")
            for range_name, count in range_counts.items():
                print(f"    - {range_name}: {count} texts ({(count/total_texts)*100:.2f}%)")
                
        except Exception as e:
            print(f"Error analyzing {csv_file}: {str(e)}")
            results[csv_file] = {'error': str(e)}
            
    return results

def plot_token_length_distribution(analysis_results, output_dir=None):
    """
    Generate plots visualizing token length distributions.
    
    Args:
        analysis_results: Results from analyze_token_lengths function
        output_dir: Directory to save plots (if None, plots are displayed but not saved)
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    for file_name, results in analysis_results.items():
        if 'error' in results:
            print(f"Skipping plot for {file_name} due to error during analysis")
            continue
            
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        
        # 1. Histogram of token counts
        plt.figure(figsize=(12, 6))
        plt.hist(results['token_counts'], bins=50, alpha=0.7, color='blue')
        plt.axvline(x=1024, color='red', linestyle='--', label='1024 token threshold')
        plt.title(f'Token Count Distribution - {base_name}')
        plt.xlabel('Token Count')
        plt.ylabel('Number of Texts')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"{base_name}_token_histogram.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # 2. Bar chart of token count ranges
        ranges = results['range_distribution']
        plt.figure(figsize=(12, 6))
        bars = plt.bar(ranges.keys(), ranges.values(), color='skyblue')
        
        # Add percentage labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            percentage = (height / results['total_texts']) * 100
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{percentage:.1f}%',
                ha='center',
                va='bottom',
                rotation=0
            )
            
        plt.title(f'Token Count Ranges - {base_name}')
        plt.xlabel('Token Count Range')
        plt.ylabel('Number of Texts')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"{base_name}_token_ranges.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        # 3. CDF of token counts
        plt.figure(figsize=(12, 6))
        sorted_counts = sorted(results['token_counts'])
        cdf = [i / len(sorted_counts) for i in range(len(sorted_counts))]
        plt.plot(sorted_counts, cdf, color='green')
        plt.axvline(x=1024, color='red', linestyle='--', label='1024 token threshold')
        
        # Find the y-value (percentile) at x=1024
        threshold_percentile = sum(count <= 1024 for count in results['token_counts']) / len(results['token_counts'])
        plt.axhline(y=threshold_percentile, color='orange', linestyle=':', 
                   label=f'{threshold_percentile*100:.1f}% of texts ≤ 1024 tokens')
        
        plt.title(f'Cumulative Distribution of Token Counts - {base_name}')
        plt.xlabel('Token Count')
        plt.ylabel('Cumulative Probability')
        plt.grid(alpha=0.3)
        plt.legend()
        
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f"{base_name}_token_cdf.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def save_token_stats_to_csv(analysis_results, output_dir="token_analysis"):
    """
    Save token analysis results to CSV files.
    
    Args:
        analysis_results: Results from analyze_token_lengths function
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary dataframe for all files
    summary_rows = []
    
    for file_name, results in analysis_results.items():
        if 'error' in results:
            continue
            
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        
        # Save detailed token counts for this file
        counts_df = pd.DataFrame({
            'token_count': results['token_counts']
        })
        counts_df.to_csv(os.path.join(output_dir, f"{base_name}_token_counts.csv"), index=False)
        
        # Add to summary
        summary_row = {
            'file_name': base_name,
            'total_texts': results['total_texts'],
            'exceeding_threshold': results['exceeding_threshold'],
            'percentage_exceeding': results['percentage_exceeding'],
            'min_tokens': results['min_tokens'],
            'mean_tokens': results['mean_tokens'],
            'median_tokens': results['median_tokens'],
            'max_tokens': results['max_tokens'],
            '90th_percentile': results['percentiles']['90th'],
            '95th_percentile': results['percentiles']['95th'],
            '99th_percentile': results['percentiles']['99th']
        }
        
        # Add range distribution
        for range_name, count in results['range_distribution'].items():
            summary_row[f'range_{range_name}'] = count
            summary_row[f'range_{range_name}_pct'] = (count / results['total_texts']) * 100
            
        summary_rows.append(summary_row)
    
    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, "token_analysis_summary.csv"), index=False)
        print(f"Saved token analysis summary to {os.path.join(output_dir, 'token_analysis_summary.csv')}")

def calculate_surprisals_sliding_window(csv_files, tokenizer_source="gpt2", max_length=2048, 
                                      window_size=1024, stride=512, output_dir="surprisal_results"):
    """
    Calculate token-level surprisals for text in CSV files using a sliding window approach
    for texts that exceed the model's context window.
    
    Args:
        csv_files: List of CSV file paths
        tokenizer_source: Either a string for HuggingFace model name (e.g., "gpt2") 
                         or a path to a tokenizer JSON file
        max_length: Maximum sequence length to consider (can be > window_size)
        window_size: Size of each processing window (1024 for GPT-2)
        stride: How much to slide the window for each step (e.g., 512 for 50% overlap)
        output_dir: Directory to save output CSV files (if None, results are not saved)
        
    Returns:
        A dictionary mapping file names to dataframes with text and their surprisals
    """
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    if tokenizer_source.endswith('.json'):
        # Load from JSON file
        with open(tokenizer_source, 'r') as f:
            tokenizer_json = json.load(f)
            tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",  # Use gpt2 as base
                vocab_file=None,
                merges_file=None,
                tokenizer_file=None
            )
            # Update with custom vocabulary
            tokenizer.vocab = tokenizer_json.get('vocab', {})
            tokenizer.ids_to_tokens = {v: k for k, v in tokenizer.vocab.items()}
    else:
        # Load from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    
    # Load a matching model based on tokenizer
    model_name = "gpt2" if tokenizer_source == "gpt2" or tokenizer_source.endswith('.json') else tokenizer_source
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    # The max window size is limited by the model's max position embeddings
    if hasattr(model.config, "max_position_embeddings"):
        model_max_length = model.config.max_position_embeddings
        if window_size > model_max_length:
            print(f"Warning: Model only supports sequences up to {model_max_length} tokens. Adjusting window_size.")
            window_size = model_max_length
    
    print(f"Using window size: {window_size}, stride: {stride}, max_length: {max_length}")
    
    results = {}
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        
        # Read the CSV
        df = pd.read_csv(csv_file)
        
        # Identify the text column - look for common text column names
        text_col = None
        possible_text_cols = ['text', 'content', 'article', 'dialogue', 'prompt', 'message', 'response']
        for col in possible_text_cols:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            # If no matching column found, use the first string column
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
                    
        if text_col is None:
            print(f"No text column found in {csv_file}, skipping.")
            continue
            
        # Create a new dataframe to store results
        result_df = pd.DataFrame()
        result_df['original_text'] = df[text_col]
        
        # Keep all original columns from the source dataframe
        for col in df.columns:
            if col != text_col:  # Skip the text column as we already have it as 'original_text'
                result_df[col] = df[col]
        
        # Store token-level surprisals
        all_surprisals = []
        all_tokens = []
        all_token_lengths = []
        
        # Process each text with progress bar
        for text in tqdm.tqdm(df[text_col], desc=f"Calculating surprisals for {os.path.basename(csv_file)}"):
            # Skip NaN or empty texts
            if pd.isna(text) or text == '':
                all_surprisals.append([])
                all_tokens.append([])
                all_token_lengths.append(0)
                continue
            
            try:
                # Tokenize the entire text first to check length
                all_token_ids = tokenizer.encode(text)
                all_token_lengths.append(len(all_token_ids))
                
                # If text is shorter than window_size, process normally
                if len(all_token_ids) <= window_size:
                    token_ids = torch.tensor([all_token_ids]).to(model.device)
                    tokens = tokenizer.convert_ids_to_tokens(all_token_ids)
                    
                    with torch.no_grad():
                        # Get model output
                        outputs = model(token_ids, labels=token_ids)
                        logits = outputs.logits
                        
                        # Shift logits and tokens for calculating next token prediction
                        shift_logits = logits[0, :-1, :].contiguous()
                        shift_tokens = token_ids[0, 1:].contiguous()
                        
                        # Get probabilities using softmax
                        probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                        
                        # Get probability of each actual next token
                        indices = torch.arange(shift_tokens.size(0))
                        token_probs = probs[indices, shift_tokens]
                        
                        # Calculate surprisal as -log(p)
                        log_probs = torch.log2(token_probs)
                        surprisals = -log_probs
                        token_surprisals = surprisals.tolist()
                        
                        # Add padding of 0 for the first token (which doesn't have a preceding context)
                        token_surprisals = [0.0] + token_surprisals
                        
                    all_surprisals.append(token_surprisals)
                    all_tokens.append(tokens)
                    
                else:
                    # For texts longer than window_size, use sliding window approach
                    # Truncate to max_length if needed
                    if len(all_token_ids) > max_length:
                        all_token_ids = all_token_ids[:max_length]
                    
                    combined_surprisals = []
                    combined_tokens = []
                    
                    # Process text in overlapping windows
                    for start_idx in range(0, len(all_token_ids), stride):
                        end_idx = min(start_idx + window_size, len(all_token_ids))
                        chunk_tokens = all_token_ids[start_idx:end_idx]
                        
                        # Skip if window is too small
                        if len(chunk_tokens) < 2:  # Need at least 2 tokens for meaningful prediction
                            continue
                            
                        chunk_ids = torch.tensor([chunk_tokens]).to(model.device)
                        chunk_text_tokens = tokenizer.convert_ids_to_tokens(chunk_tokens)
                        
                        with torch.no_grad():
                            # Get model output for this chunk
                            outputs = model(chunk_ids, labels=chunk_ids)
                            logits = outputs.logits
                            
                            # Shift logits and tokens for calculating next token prediction
                            shift_logits = logits[0, :-1, :].contiguous()
                            shift_tokens = chunk_ids[0, 1:].contiguous()
                            
                            # Get probabilities using softmax
                            probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                            
                            # Get probability of each actual next token
                            indices = torch.arange(shift_tokens.size(0))
                            token_probs = probs[indices, shift_tokens]
                            
                            # Calculate surprisal as -log(p)
                            log_probs = torch.log2(token_probs)
                            surprisals = -log_probs
                            chunk_surprisals = surprisals.tolist()
                            
                            # Add padding of 0 for the first token in chunk
                            chunk_surprisals = [0.0] + chunk_surprisals
                        
                        # For the first window, use all surprisals
                        if start_idx == 0:
                            combined_surprisals.extend(chunk_surprisals)
                            combined_tokens.extend(chunk_text_tokens)
                        else:
                            # For subsequent windows, only use surprisals for tokens beyond overlap
                            # The last (stride) tokens from the previous window overlap with the 
                            # first (stride) tokens of this window
                            keep_from_idx = stride
                            
                            # If we've reached the end of the text, we might need fewer tokens
                            if end_idx >= len(all_token_ids):
                                keep_from_idx = min(keep_from_idx, len(chunk_surprisals) - (end_idx - start_idx))
                                
                            # Only add the new tokens (those beyond the overlap)
                            if keep_from_idx < len(chunk_surprisals):
                                combined_surprisals.extend(chunk_surprisals[keep_from_idx:])
                                combined_tokens.extend(chunk_text_tokens[keep_from_idx:])
                        
                        # If we've reached the end of the text, stop
                        if end_idx >= len(all_token_ids):
                            break
                    
                    all_surprisals.append(combined_surprisals)
                    all_tokens.append(combined_tokens)
                    
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                # Return empty lists for failures
                all_surprisals.append([])
                all_tokens.append([])
        
        # Add surprisals to result dataframe
        result_df['token_surprisals'] = all_surprisals
        result_df['tokens'] = all_tokens
        result_df['token_length'] = all_token_lengths
        
        # Calculate average surprisal per text
        result_df['avg_surprisal'] = result_df['token_surprisals'].apply(
            lambda x: np.mean(x) if len(x) > 0 else np.nan
        )
        
        # Create token-level dataframe for detailed analysis
        token_rows = []
        for i, (surprisals, tokens, text) in enumerate(zip(all_surprisals, all_tokens, result_df['original_text'])):
            for j, (token, surprisal) in enumerate(zip(tokens, surprisals)):
                if j < len(surprisals):  # Ensure we don't go out of bounds
                    row = {
                        'text_id': i,
                        'token_id': j,
                        'token': token,
                        'surprisal': surprisal,
                        'text': text
                    }
                    token_rows.append(row)
        
        token_df = pd.DataFrame(token_rows)
        
        # Store both dataframes in the results dictionary
        results[csv_file] = {
            'text_level': result_df,
            'token_level': token_df
        }
        
        # Save to CSV if output_dir is provided
        if output_dir is not None:
            # Get the base filename without extension
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Save text-level dataframe
            # Convert token_surprisals list to string for CSV storage
            result_df_for_csv = result_df.copy()
            result_df_for_csv['token_surprisals'] = result_df_for_csv['token_surprisals'].apply(
                lambda x: ','.join([str(val) for val in x]) if len(x) > 0 else ''
            )
            result_df_for_csv['tokens'] = result_df_for_csv['tokens'].apply(
                lambda x: ','.join([str(val) for val in x]) if len(x) > 0 else ''
            )
            
            result_df_for_csv.to_csv(os.path.join(output_dir, f"{base_name}_text_surprisals.csv"), index=False)
            
            # Save token-level dataframe
            token_df.to_csv(os.path.join(output_dir, f"{base_name}_token_surprisals.csv"), index=False)
            
            print(f"Saved results for {csv_file} to {output_dir}")
    
    return results

def analyze_sliding_window_results(results, output_dir=None):
    """
    Analyze and print statistics about the sliding window surprisal results.
    
    Args:
        results: Results dictionary from calculate_surprisals_sliding_window
        output_dir: Directory where additional analysis files can be saved
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
    summary_rows = []
    
    for file_name, data in results.items():
        base_name = os.path.basename(file_name)
        text_df = data['text_level']
        token_df = data['token_level']
        
        # Filter out rows with no valid surprisal values
        valid_surprisals = text_df['avg_surprisal'].dropna()
        
        # Token length distribution
        token_lengths = text_df['token_length'].tolist()
        tokens_exceeding_1024 = sum(1 for length in token_lengths if length > 1024)
        tokens_exceeding_2048 = sum(1 for length in token_lengths if length > 2048)
        
        # Print summary
        print(f"\n=== Summary for {base_name} ===")
        print(f"Total texts: {len(text_df)}")
        print(f"Texts with valid surprisals: {len(valid_surprisals)}")
        print(f"Total tokens analyzed: {len(token_df)}")
        
        if len(valid_surprisals) > 0:
            print(f"Surprisal statistics:")
            print(f"  - Average surprisal: {valid_surprisals.mean():.4f}")
            print(f"  - Min surprisal: {valid_surprisals.min():.4f}")
            print(f"  - Max surprisal: {valid_surprisals.max():.4f}")
            
        print(f"Token length statistics:")
        print(f"  - Average token length: {np.mean(token_lengths):.2f}")
        print(f"  - Median token length: {np.median(token_lengths):.2f}")
        print(f"  - Max token length: {max(token_lengths) if token_lengths else 0}")
        print(f"  - Texts exceeding 1024 tokens: {tokens_exceeding_1024} ({tokens_exceeding_1024/len(text_df)*100:.2f}%)")
        print(f"  - Texts exceeding 2048 tokens: {tokens_exceeding_2048} ({tokens_exceeding_2048/len(text_df)*100:.2f}%)")
        
        # Save detailed token length distribution if output_dir provided
        if output_dir is not None:
            # Create token length histogram data
            bins = [0, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096, float('inf')]
            bin_labels = ['0-256', '257-512', '513-768', '769-1024', '1025-1536', 
                          '1537-2048', '2049-2560', '2561-3072', '3073-4096', '4096+']
            
            hist, _ = np.histogram(token_lengths, bins=bins)
            hist_df = pd.DataFrame({
                'range': bin_labels,
                'count': hist,
                'percentage': [count/len(text_df)*100 for count in hist]
            })
            
            hist_df.to_csv(os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_token_length_distribution.csv"), 
                           index=False)
            
        # Collect summary data
        summary_row = {
            'file_name': base_name,
            'total_texts': len(text_df),
            'valid_surprisals': len(valid_surprisals),
            'total_tokens': len(token_df),
            'avg_surprisal': valid_surprisals.mean() if len(valid_surprisals) > 0 else np.nan,
            'min_surprisal': valid_surprisals.min() if len(valid_surprisals) > 0 else np.nan,
            'max_surprisal': valid_surprisals.max() if len(valid_surprisals) > 0 else np.nan,
            'avg_token_length': np.mean(token_lengths) if token_lengths else 0,
            'median_token_length': np.median(token_lengths) if token_lengths else 0,
            'max_token_length': max(token_lengths) if token_lengths else 0,
            'tokens_exceeding_1024': tokens_exceeding_1024,
            'pct_exceeding_1024': tokens_exceeding_1024/len(text_df)*100 if len(text_df) > 0 else 0,
            'tokens_exceeding_2048': tokens_exceeding_2048,
            'pct_exceeding_2048': tokens_exceeding_2048/len(text_df)*100 if len(text_df) > 0 else 0,
        }
        
        summary_rows.append(summary_row)
    
    # Save summary table if output_dir provided
    if output_dir is not None and summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(output_dir, "sliding_window_analysis_summary.csv"), index=False)
        print(f"\nSaved analysis summary to {os.path.join(output_dir, 'sliding_window_analysis_summary.csv')}")
    
    return summary_rows