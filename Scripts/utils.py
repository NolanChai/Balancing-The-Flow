import numpy as np
from bs4 import BeautifulSoup
import re 
import time
import random
import requests
import csv
import scrapy # may or may not try with this
import pandas as pd
import json
import torch
from datasets import load_dataset
from urllib.parse import urljoin, urlparse
import nltk
from tqdm import tqdm
from pathlib import Path
import os
import glob
import sys

nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def is_valid(url):
    parsed = urlparse(url)
    valid_connect = bool(parsed.netloc)
    valid_scheme = bool(parsed.scheme)
    return valid_connect and valid_scheme

def get_title(article):
    return article.find('h2', class_="headline").text

def get_first_sentence(article):
    article = article.strip()
    sentences = sent_tokenize(article)
    return sentences[0] if sentences else article

# Code borrowed from Dr. Alex Warstadt
def to_tokens_and_logprobs(model, tokenizer, input_texts, disable_progress=False, quiet=False):
    """
    Calculate token-level surprisals for input texts
    
    Args:
        model: The language model to use
        tokenizer: The tokenizer for the model
        input_texts: List of texts to process
        disable_progress: Whether to disable progress bars
        quiet: Whether to suppress all print statements
    
    Returns:
        List of dataframes containing token and surprisal information
    """
    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize inputs
    input_ids = tokenizer(input_texts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    
    # Run model inference
    if not quiet:
        print("Running model inference...", end="", flush=True)
    t0 = time.time()
    outputs = model(input_ids)
    t1 = time.time()
    model.to("cpu")  # Free up GPU memory
    if not quiet:
        print(f" done in {t1-t0:.2f}s", flush=True)

    # Calculate surprisals
    if not quiet:
        print("Computing surprisals...", end="", flush=True)
    t0 = time.time()  
    logits = outputs.logits.cpu().detach()
    probs = torch.softmax(logits, dim=-1)
    surprisals = -1 * torch.log2(probs)

    # Align tokens and surprisals
    input_ids = input_ids.cpu().detach()
    surprisals = surprisals[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_surprisals = torch.gather(surprisals, 2, input_ids[:, :, None]).squeeze(-1)
    t1 = time.time()
    if not quiet:
        print(f" done in {t1-t0:.2f}s", flush=True)
    
    # Create dataframes
    batch = []
    for i, id_surp in enumerate(zip(input_ids, gen_surprisals)):
        sentence = []
        input_sentence, input_surprisals = id_surp
        for token, p in zip(input_sentence, input_surprisals):
            if token not in tokenizer.all_special_ids:
                sentence.append({
                    "token": tokenizer.decode(token),
                    "surprisal": p.item()
                })
        batch.append(pd.DataFrame(sentence))
    
    return batch

def scrape(article_urls):
    """
    Returns a list of all the soups extracted from url, or list of urls
    """
    pot = []
    if isinstance(article_urls, str):
        if not is_valid(url):
            return Exception
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        pot.add(soup)
    else:
        for i, url in enumerate(article_urls):
            if not is_valid(url):
                continue
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')
            pot.add(soup)
    return pot

def prompt_hfds(num_articles, client, temperature, model_name, dataset_name,
                model_output_dir="../Generations", human_output_dir="../Sources", 
                write_source=True, regenerate=True, max_tokens=2048, top_p=1.0, 
                system_prompt=None, max_retries=3, dataset_config="", source_only=False):
    """Prompts an LM using articles from the cnn_dailymail dataset

    Args:
        num_articles (int): number of articles to generate
        client : API client used for model prompting
        temperature (float): temperature parameter for the model
        model_name (str): name of the model
        dataset_name (str): name of the dataset to prompt from
        model_output_dir (str, optional): output directory for generations. Defaults to "../Generations".
        human_output_dir (str, optional): output directory for human articles. Defaults to "../Sources".
        write_source (bool, optional): if True, write the human articles to .txt files in human_output_dir. Defaults to True.
        regenerate (bool, optional): if True, regenerate articles that already exist. Defaults to True.
        max_tokens (int, optional): maximum tokens for generation. Defaults to 2048.
        max_retries (int, optional): maximum number of retries for failed generations. Defaults to 3.
        dataset_config (str, optional): configuration for dataset
        source_only (bool, optional): if True, skip generation and only write source files
    """
    MIN_GENERATION_LENGTH = 200
    # handling for missing paths
    model_output_path = Path(model_output_dir) / Path(model_name) / Path(dataset_name.split("/")[-1])
    human_output_path = Path(human_output_dir) / Path(dataset_name.split("/")[-1])

    model_output_path.mkdir(parents=True, exist_ok=True)
    if write_source:
        human_output_path.mkdir(parents=True, exist_ok=True)

    dataset_params = (dataset_name,)
    if dataset_config:
        dataset_params += (dataset_config,)
    data = load_dataset(*dataset_params, trust_remote_code=True)['test']
    shuffled_data = data.shuffle(seed=42)  # Use consistent seed for reproducibility

    # setting articles to generate
    if num_articles < 0:
        num_articles = len(shuffled_data)

    # make sure it doesn't exceed dataset size
    num_articles = min(num_articles, len(shuffled_data))
    
    # Configure progress bar with better settings
    pbar = tqdm(
        total=num_articles,
        desc=f"Generating with {model_name}",
        unit="articles",
        position=0,
        leave=True
    )
    
    generated_count = 0
    retry_count = 0
    skip_count = 0
    error_count = 0

    for i in range(num_articles):
        # output paths
        generation_output_path = model_output_path / f"{model_name}_{i}.txt"
        source_output_path = human_output_path / f"human_{i}.txt"

        # skip if exists
        if not regenerate and generation_output_path.exists():
            pbar.update(1)
            skip_count += 1
            continue
                        
        # write source article if needed
        if (not source_output_path.exists() or regenerate) and write_source:
            with open(source_output_path, "w", encoding="utf-8") as outfile:
                outfile.write(get_text(dataset=dataset_name, item=shuffled_data[i]))

        # try up to max retries
        for retry in range(max_retries):
            # skip generation if source only
            if source_only:
                break
            try:
                prompt = get_prompt(
                    dataset=dataset_name, 
                    item=shuffled_data[i])
                # print("Prompt:", prompt)
                # generate text
                generation = generate(
                    client=client, 
                    prompt=prompt, 
                    temperature=temperature, 
                    model=model_name,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    system_prompt=system_prompt
                )
                
                # validate generation length
                if len(generation.strip()) < MIN_GENERATION_LENGTH:
                    if retry < max_retries - 1:
                        retry_count += 1
                        time.sleep(1)  # Wait briefly before retry
                        continue
                    else:
                        # Last retry, use what we got
                        generation = f"[Warning: Short generation] {generation}"
                
                # write to path
                with open(generation_output_path, "w", encoding="utf-8") as outfile:
                    outfile.write(generation)
                
                generated_count += 1
                break  # success!
                
            except Exception as e:
                error_msg = str(e)
                if retry < max_retries - 1:
                    retry_count += 1
                    time.sleep(1)  # wait a bit before trying again
                else:
                    # retry failed
                    error_count += 1
                    print(f"\nError processing text {i} after {max_retries} attempts: {error_msg[:100]}...", file=sys.stderr)
                    
                    # error placeholder
                    try:
                        with open(generation_output_path, "w", encoding="utf-8") as outfile:
                            outfile.write(f"[ERROR] Generation failed after {max_retries} attempts: {error_msg[:100]}...")
                    except:
                        pass
        pbar.update(1)
        
        if i % 50 == 0 and i > 0:
            pbar.set_description(f"Gen: {generated_count}, Skip: {skip_count}, Retry: {retry_count}, Err: {error_count}")
    
    pbar.close()
    
    print(f"Generation complete: {generated_count} new, {skip_count} skipped, {error_count} errors, {retry_count} retries")
    return generated_count, skip_count, error_count, retry_count

def get_text(dataset, item):
    if dataset == "abisee/cnn_dailymail":
        text = item['article']
    elif dataset == "euclaise/writingprompts":
        text = item['story']
    else:
        raise NotImplementedError(f"Dataset {dataset} not yet supported. Please specify prompting method.")
    return text

def get_prompt(dataset, item, min_prompt_len=200, max_prompt_len=500):
    text = get_text(dataset, item)
    if dataset in ["abisee/cnn_dailymail",
                   "euclaise/writingprompts",
                   ]:
        # use first sentence as prompt
        prompt = get_first_sentence(text)
    else:
        raise NotImplementedError(f"Dataset {dataset} not yet supported. Please specify prompting method.")
    if (len(prompt.strip()) < min_prompt_len) or (len(prompt.strip()) > max_prompt_len):
            prompt = text[:min_prompt_len]  # Take first 200 chars if prompt is too short/long
    return prompt
    
def generate(client, prompt, temperature, model, max_tokens=2048, top_p=1.0, system_prompt=None):
    """Generate a completion using the given API client, model and parameters

    Args:
        client : API client used for model prompting
        prompt (str): prompt to give the model
        temperature (float): temperature parameter for the model
        model (str): name of the model
        max_tokens (int, optional): maximum number of tokens to generate. Defaults to 2048.
        top_p (float, optional): nucleus sampling parameter. Defaults to 1.0.
        system_prompt (str, optional): system prompt to prepend. Defaults to None.

    Returns:
        str: generated completion
    """
    # message list
    messages = []
    
    # add prompts
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # add user prompt
    messages.append({"role": "user", "content": prompt})
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    
    return completion.choices[0].message.content

def split_batches(lst, batch_size):
    """splits a given list into batches of a given size

    Args:
        lst (list): list to batch
        batch_size (int): size of batches

    Returns:
        list: list of batches of size batch_size
    """
    num_batches = int(np.ceil(len(lst) / batch_size))
    batched = []
    for i in range(num_batches):
        start_idx = i * batch_size
        batch = lst[start_idx:start_idx + batch_size]
        batched.append(batch)
    return batched

def calc_surprisal(model, tokenizer, input_dir, output_dir, model_name, num_files=-1, batch_size=20, verbose=False):
    """Calculate surprisal for each token of a corpus of texts

    Args:
        model (GPT2LMHeadModel): pretrained language model to use for probability calculation
        tokenizer (GPT2Tokenizer): tokenizer associated with the model
        input_dir (str/path): path to directory containing corpus
        output_dir (str/path): desired output directory for csv files
        model_name (str): name of the model to filter files for
        num_files (int, optional): number of files to analyze. Passing -1 will analyze all files in the directory.
        batch_size (int, optional): batch size for surprisal calculation input. Defaults to 20.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    root = Path(input_dir)
    
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    file_pattern = f"{model_name}_*.txt"
    text_filepaths = list(root.glob(file_pattern))
    
    if not text_filepaths:
        raise ValueError(f"No text files found in {input_dir} matching pattern {file_pattern}")
    
    text_filepaths.sort()
    
    absolute_filepaths = [file.resolve() for file in text_filepaths]
    
    if num_files > 0:
        absolute_filepaths = absolute_filepaths[:min(num_files, len(absolute_filepaths))]
    
    if not absolute_filepaths:
        print(f"No files to process in {input_dir} matching {file_pattern}")
        return
    
    if verbose:
        print(f"Reading {len(absolute_filepaths)} files from {input_dir} for model {model_name}")
    
    all_texts = []
    file_paths_to_use = []
    
    file_pbar = tqdm(
        absolute_filepaths,
        desc=f"Reading {model_name} files",
        unit="file",
        disable=not verbose,
        position=0,
        leave=True
    )
    
    for filepath in file_pbar:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            if text.strip():
                all_texts.append(text)
                file_paths_to_use.append(filepath)
        except Exception as e:
            print(f"Error reading file {filepath.name}: {e}")
    
    file_pbar.close()
    
    if not all_texts:
        print("No valid text content found to analyze")
        return
    
    if verbose:
        print(f"Successfully read {len(all_texts)} files")
    
    batched_texts = split_batches(all_texts, batch_size=batch_size)
    batched_filepaths = split_batches(file_paths_to_use, batch_size=batch_size)
    
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    batch_pbar = tqdm(
        total=len(batched_texts),
        desc=f"Processing {model_name} batches",
        unit="batch",
        disable=not verbose,
        position=0,
        leave=True
    )
    
    for batch_idx, (texts, paths) in enumerate(zip(batched_texts, batched_filepaths)):
        try:
            if verbose:
                print(f"\nProcessing batch {batch_idx+1}/{len(batched_texts)} with {len(texts)} files")
            
            outputs = to_tokens_and_logprobs(model, tokenizer, texts, disable_progress=True, quiet=True)
            
            for fp, df in zip(paths, outputs):
                try:
                    output_fp = output_root / f"{fp.stem}.csv"
                    df.to_csv(output_fp, index=False)
                except Exception as e:
                    print(f"Error writing file {fp.name}: {e}")
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}: {e}")
        finally:
            batch_pbar.update(1)
    
    batch_pbar.close()
            
    if verbose:
        print(f"Surprisal calculation complete for {model_name}. Results saved to {output_dir}")

# trying this out
def modified_to_tokens_and_logprobs(model, tokenizer, input_texts, verbose=False):
    """A modified version of to_tokens_and_logprobs with better error handling
    and explicit padding setting
    """
    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    model.to(device)

    # padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if verbose:
            print("Set padding token to end-of-sequence token")

    encoded = tokenizer(
        input_texts, 
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded.input_ids.to(device)

    if verbose:
        print("Running through model:")
    t0 = time.time()

    # trying no_grad for efficiency; we don't need gradients since we're just calculating surprisals w log likelihoods
    with torch.no_grad():
        outputs = model(input_ids)

    # testing
    t1 = time.time()
    if verbose:
        print(f"Model inference time: {t1 - t0:.2f} seconds")

    model.to("cpu") # moving to cpu

    if verbose:
        print("Calculating surprisals")
    t0 = time.time()

    logits = outputs.logits.cpu().detach()
    probs = torch.softmax(logits, dim=-1)
    surprisals = -1 * torch.log2(probs)

    surprisals = surprisals[:, :-1, :]
    actual_tokens = input_ids[:, 1:].cpu().detach()
    token_surprisals = torch.gather(surprisals, 2, actual_tokens[:, :, None]).squeeze(-1)

    t1 = time.time()
    if verbose:
        print(f"Surprisal calculation time: {t1 - t0:.2f} seconds")

    batch_results = []
    for i, (input_tensor, surprisal_tensor) in enumerate(zip(actual_tokens, token_surprisals)):
        tokens_data = []
        for j, (token_id, surprisal) in enumerate(zip(input_tensor, surprisal_tensor)):
            if token_id not in tokenizer.all_special_ids:  # Skip special tokens
                token_text = tokenizer.decode(token_id)
                tokens_data.append({
                    "token": token_text,
                    "surprisal": surprisal.item()
                })
        
        # create df for output here
        batch_results.append(pd.DataFrame(tokens_data))
    
    return batch_results

def calculate_surprisals_for_existing_texts(input_dir, output_dir, model, tokenizer, model_name="human", pattern=None, verbose=False):
    """Calculate surprisals for existing text files
    
    Args:
        input_dir (str): Directory containing text files
        output_dir (str): Directory to save surprisal CSVs
        model: The language model to use
        tokenizer: The tokenizer for the model
        model_name (str): Name to identify the source (e.g., "human" or model name)
        pattern (str): Pattern to match files. If None, uses "{model_name}_*.txt"
        verbose (bool): Whether to print detailed output
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up file pattern for matching
    if pattern is None:
        pattern = f"{model_name}_*.txt"
    
    # Get all text files matching the pattern
    input_path = Path(input_dir)
    text_files = list(input_path.glob(pattern))
    
    if len(text_files) == 0:
        print(f"No text files found in {input_dir} matching pattern {pattern}")
        return
        
    if verbose:
        print(f"Found {len(text_files)} files in {input_dir} matching {pattern}")
    
    # Sort files to ensure consistent processing
    text_files.sort()
    
    # Process files
    all_texts = []
    filepaths = []
    
    # Read all files first
    for filepath in tqdm(text_files, desc=f"Reading {model_name} files", disable=not verbose):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            if text.strip():
                all_texts.append(text)
                filepaths.append(filepath)
        except Exception as e:
            if verbose:
                print(f"Error reading {filepath}: {e}")
    
    if len(all_texts) == 0:
        print("No valid text content found to analyze")
        return
        
    if verbose:
        print(f"Successfully read {len(all_texts)} files for {model_name}")
    
    # Process in batches
    batch_size = 20
    
    # Split into batches
    batched_texts = split_batches(all_texts, batch_size)
    batched_filepaths = split_batches(filepaths, batch_size)
    
    # Calculate surprisals
    output_root = Path(output_dir)
    total_processed = 0
    total_batches = len(batched_texts)
    
    # Use tqdm.auto for better terminal compatibility
    with tqdm(total=total_batches, desc=f"Processing {model_name} batches", disable=not verbose) as pbar:
        for batch_idx, (texts, paths) in enumerate(zip(batched_texts, batched_filepaths)):
            try:
                # Use quiet mode to prevent output during batch processing
                outputs = to_tokens_and_logprobs(model, tokenizer, texts, quiet=True)
                
                # Save CSV files
                for path, df in zip(paths, outputs):
                    output_path = output_root / f"{path.stem}.csv"
                    df.to_csv(output_path, index=False)
                
                total_processed += len(texts)
                
                # Update progress bar description with counters
                if verbose:
                    pbar.set_description(f"Processing {model_name} batch {batch_idx+1}/{total_batches} ({total_processed}/{len(all_texts)} files)")
                
            except Exception as e:
                if verbose:
                    pbar.write(f"Error processing batch {batch_idx+1}: {e}")
            
            # Update progress bar
            pbar.update(1)
    
    # Output final stats
    if verbose:
        print(f"Surprisal calculation complete for {model_name}. Results saved to {output_dir}")
        print(f"Processed {total_processed} files")
        
        # Count actual output files
        output_files = list(output_root.glob("*.csv"))
        print(f"Generated {len(output_files)} surprisal CSV files for {model_name}")