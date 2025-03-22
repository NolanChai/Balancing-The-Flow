from openai import OpenAI
from pathlib import Path
from bs4 import BeautifulSoup
from utils import prompt_hfds, generate, calc_surprisal, get_first_sentence
import requests
from datasets import load_dataset
import argparse
import numpy as np
import sys
import re
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd

def setup_directories(model_name):
    """
    Create necessary directories for output
    """
    dirs = {
        "generations": Path("../Generations"),
        "sources": Path("../Sources"),
        "surprisals": Path(f"../Surprisals/{model_name}")
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured {name} directory exists at: {path}")
    
    return dirs

def calculate_average_surprisal(model_name, num_files):
    """
    Calculate average surprisal across generated files
    """

    surprisal_dir = Path(f"../Surprisals/{model_name}")
    surprisal_dir.mkdir(parents=True, exist_ok=True)

    # check for surprisals
    existing_csvs = list(surprisal_dir.glob("*.csv"))
    if len(existing_csvs) < num_files:
        if verbose:
            print(f"Calculating surprisals for {model_name}...")
        
        # load model
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            
            # calc
            calc_surprisal(
                model=model,
                tokenizer=tokenizer,
                input_dir=f"../Generations",
                output_dir=str(surprisal_dir),
                num_files=num_files
            )
        except Exception as e:
            print(f"Error calculating surprisals: {e}")
            return None
    
    # try from csv files
    try:
        surprisal_files = list(surprisal_dir.glob("*.csv"))
        if not surprisal_files:
            print("No surprisal files found")
            return None
            
        total_surprisal = 0
        total_tokens = 0
        
        if verbose:
            print(f"Calculating average surprisal from {len(surprisal_files)} files...")
            file_iter = tqdm(surprisal_files)
        else:
            file_iter = surprisal_files
        
        for file_path in file_iter:
            df = pd.read_csv(file_path)
            if 'surprisal' in df.columns:
                total_surprisal += df['surprisal'].sum()
                total_tokens += len(df)
        
        avg_surprisal = total_surprisal / total_tokens if total_tokens > 0 else 0
        if verbose:
            print(f"Average surprisal across {len(surprisal_files)} files: {avg_surprisal:.4f}")
            print(f"Total tokens processed: {total_tokens}")
        
        return avg_surprisal
    
    except Exception as e:
        print(f"Error in average surprisal calculation: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using language models via LM Studio API')
    parser.add_argument('model', type=str, help='Model name to use for generation')
    parser.add_argument('-g', '--generate', type=int, default=300, help='Number of examples to generate')
    parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('-r', '--regenerate', action='store_true', help='Regenerate existing outputs')
    parser.add_argument('--verbose', type=bool, default=False, help='Print surprisal information')

    # parse known
    args, unknown = parser.parse_known_args()

    extra_args = {}
    for arg in unknown:
        # edge case handling
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert value to appropriate type
            if value.lower() == 'true':
                extra_args[key] = True
            elif value.lower() == 'false':
                extra_args[key] = False
            elif value.isdigit():
                extra_args[key] = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                extra_args[key] = float(value)
            else:
                extra_args[key] = value
    verbose = extra_args.get('verbose', args.verbose)

    HOST = "http://localhost:1234/v1"
    CLIENT = OpenAI(base_url=HOST, api_key="lm-studio")

    if verbose:
        print(f"Using model: {args.model}")
        print(f"Generating {args.generate} examples")
        print(f"Temperature: {args.temperature}")
        print(f"Regenerate: {args.regenerate}")

    prompt_hfds(
        num_articles=args.generate, 
        client=CLIENT, 
        temperature=args.temperature, 
        model_name=args.model, 
        regenerate=args.regenerate
    )

    if verbose:
        avg_surprisal = calculate_average_surprisal(args.model, args.generate)
        print(f"Generation complete. Average surprisal: {avg_surprisal}")