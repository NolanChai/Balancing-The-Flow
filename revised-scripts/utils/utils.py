import os
import re
import sys
import json
import datetime
import pandas as pd
from pathlib import Path
from utils.spinner import Spinner

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