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
import time

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

def calculate_average_surprisal(model_name, num_files, verbose=False):
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

            # padding issue fix? if missing padding, set to eos token if dne
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                if verbose:
                    print("Set padding token to end-of-sequence token")

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

def main():
    parser = argparse.ArgumentParser(description='Generate text using language models via LM Studio API')
    parser.add_argument('model', type=str, help='Model name to use for generation')
    parser.add_argument('-g', '--generate', type=int, default=300, help='Number of examples to generate')
    parser.add_argument('-t', '--temperature', type=float, default=0.9, help='Temperature for generation')
    parser.add_argument('-p', '--top-p', type=float, default=1.0, help='Top-p (nucleus sampling) parameter')
    parser.add_argument('-r', '--regenerate', action='store_true', help='Regenerate existing outputs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose information')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Maximum tokens for generation')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries for failed generations')

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

    top_p = extra_args.get('top_p', args.top_p)

    HOST = "http://localhost:1234/v1"
    CLIENT = OpenAI(base_url=HOST, api_key="lm-studio")

    if verbose:
        print(f"Using model: {args.model}")
        print(f"Generating {args.generate} examples")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {top_p}")
        print(f"Regenerate: {args.regenerate}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Max retries: {args.max_retries}")
        print(f"Starting generation at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        generated, skipped, errors, retries = prompt_hfds(
            num_articles=args.generate, 
            client=CLIENT, 
            temperature=args.temperature, 
            top_p=top_p,
            model_name=args.model, 
            regenerate=args.regenerate,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    generation_time = time.time() - start_time
    if verbose:
        print(f"Generation completed in {generation_time:.2f} seconds")

    if verbose:
        print("Calculating surprisal statistics...")
        surprisal_start = time.time()
        avg_surprisal = calculate_average_surprisal(args.model, args.generate, verbose)
        surprisal_time = time.time() - surprisal_start
        
        print("\nSummary:")
        print(f"- Model: {args.model}")
        print(f"- Generated: {generated} new examples")
        print(f"- Skipped: {skipped} existing examples")
        print(f"- Errors: {errors} failed generations")
        print(f"- Retries: {retries} retried generations")
        print(f"- Generation time: {generation_time:.2f} seconds")
        print(f"- Surprisal calculation time: {surprisal_time:.2f} seconds")
        if avg_surprisal is not None:
            print(f"- Average surprisal: {avg_surprisal:.4f}")
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")

        try:
            summary_dir = Path("../Summary")
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            summary_file = summary_dir / f"{args.model}_summary.txt"
            with open(summary_file, "w") as f:
                f.write(f"Model: {args.model}\n")
                f.write(f"Generated: {generated} new examples\n")
                f.write(f"Skipped: {skipped} existing examples\n")
                f.write(f"Errors: {errors} failed generations\n")
                f.write(f"Retries: {retries} retried generations\n")
                f.write(f"Temperature: {args.temperature}\n")
                f.write(f"Generation time: {generation_time:.2f} seconds\n")
                f.write(f"Surprisal calculation time: {surprisal_time:.2f} seconds\n")
                if avg_surprisal is not None:
                    f.write(f"Average surprisal: {avg_surprisal:.4f}\n")
                f.write(f"Total runtime: {time.time() - start_time:.2f} seconds\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"Summary saved to {summary_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")

if __name__ == "__main__":
    main()