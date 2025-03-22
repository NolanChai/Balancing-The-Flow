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
def to_tokens_and_logprobs(model, tokenizer, input_texts):
    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer(input_texts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    print("Running through model:")
    t0 = time.time()
    outputs = model(input_ids)
    t1 = time.time()
    model.to("cpu")
    print(f"Total time: {t1 - t0}")

    print("Calculating surprisals")
    t0 = time.time()  
    logits = outputs.logits.cpu().detach()
    probs = torch.softmax(logits, dim=-1)
    # probs = probs.cpu().detach()
    surprisals = -1 * np.log2(probs)

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    input_ids = input_ids.cpu().detach()
    surprisals = surprisals[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_surprisals = torch.gather(surprisals, 2, input_ids[:, :, None]).squeeze(-1)
    t1 = time.time()
    print(f"Total time: {t1 - t0}")
    # gather all the surprisals for the sequences into a neat table
    print("Creating surprisal tables:")
    batch = []
    sentence_id = 0
    for i, id_surp in tqdm(enumerate(zip(input_ids, gen_surprisals))):
        sentence = []
        input_sentence, input_surprisals = id_surp
        for token, p in zip(input_sentence, input_surprisals):
            if token not in tokenizer.all_special_ids:
                sentence.append({
                    # "sentence_id": sentence_id,
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

def prompt_hfds(num_articles,client, temperature, model_name, 
                model_output_dir="../Generations", human_output_dir="../Sources", 
                write_source=True, regenerate=True, max_tokens=2048, top_p=1.0, max_retries=3):
    """Prompts an LM using articles from the cnn_dailymail dataset

    Args:
        num_articles (int): number of articles to generate
        client : API client used for model prompting
        temperature (float): temperature parameter for the model
        model_name (str): name of the model
        model_output_dir (str, optional): output directory for generations. Defaults to "../Generations".
        human_output_dir (str, optional): output directory for human articles. Defaults to "../Sources".
        write_source (bool, optional): if True, write the human articles to .txt files in human_output_dir. Defaults to True.
        regenerate (bool, optional): if True, regenerate articles that already exist. Defaults to True.
        max_tokens (int, optional): maximum tokens for generation. Defaults to 2048.
        max_retries (int, optional): maximum number of retries for failed generations. Defaults to 3.
    """

    # handling for missing paths
    model_output_path = Path(model_output_dir)
    human_output_path = Path(human_output_dir)

    model_output_path.mkdir(parents=True, exist_ok=True)
    if write_source:
        human_output_path.mkdir(parents=True, exist_ok=True)

    data = load_dataset("abisee/cnn_dailymail", "1.0.0", trust_remote_code=True)['train']
    shuffled_data = data.shuffle(seed=42)  # Use consistent seed for reproducibility

    # setting articles to generate
    if num_articles < 0:
        num_articles = 10_000

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

        # try up to max retries
        for retry in range(max_retries):
            try:
                # get article & first sentence
                article = shuffled_data[i]
                first_sent = get_first_sentence(article['article'])
                
                if len(first_sent.strip()) < 10:
                    first_sent = article['article'][:200]  # Take first 200 chars if first sentence is too short
                
                # generate text
                generation = generate(
                    client=client, 
                    prompt=first_sent, 
                    temperature=temperature, 
                    model=model_name,
                    max_tokens=max_tokens,
                    top_p=top_p  # Pass the top_p parameter
                )
                
                # validate generation length
                if len(generation.strip()) < 20:
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
                
                # write source article if needed
                if (not source_output_path.exists() or regenerate) and write_source:
                    with open(source_output_path, "w", encoding="utf-8") as outfile:
                        outfile.write(article['article'])
                
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
                    print(f"\nError processing article {i} after {max_retries} attempts: {error_msg[:100]}...", file=sys.stderr)
                    
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


def generate(client, prompt, temperature=0.9, model="llama-2-7b@q8_0", max_tokens=2048, top_p=0.6):
    """Generate a completion using the given API client, model and temperature parameter
    with additional validation to prevent abnormal outputs.

    Args:
        client : API client used for model prompting
        prompt (str): prompt to give the model
        temperature (float): temperature parameter for the model
        model (str): name of the model
        max_tokens (int, optional): maximum number of tokens to generate. Defaults to 2048.
        max_length (int, optional): maximum length of output in characters. Defaults to 5000.
        max_newlines (int, optional): maximum number of consecutive newlines allowed. Defaults to 20.

    Returns:
        str: generated completion, sanitized and validated
    """
    completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
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

def calc_surprisal(model, tokenizer, input_dir, output_dir, num_files=-1, batch_size=20, verbose=False):
    """Calculate surprisal for each token of a corpus of texts

    Args:
        model (GPT2LMHeadModel): pretrained language model to use for probability calculation
        tokenizer (GPT2Tokenizer): tokenizer associated with the model
        input_dir (str/path): path to directory containing corpus
        output_dir (str/path): desired output directory for csv files
        num_files (int, optional): number of files to analyze. Passing -1 will analyze all files in the directory.
        batch_size (int, optional): batch size for surprisal calculation input. Defaults to 20.
    """
    
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    root = Path(input_dir)
    
    # file handling
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
        
    text_filepaths = list(root.glob("*.txt"))
    if not text_filepaths:
        raise ValueError(f"No text files found in {input_dir}")
        
    absolute_filepaths = [file.resolve() for file in text_filepaths]

    # more file handling
    if num_files > 0:
        absolute_filepaths = absolute_filepaths[:min(num_files, len(absolute_filepaths))]
    
    if not absolute_filepaths:
        print(f"No files to process in {input_dir}")
        return
    
    if verbose:
        print(f"Reading {len(absolute_filepaths)} files from {input_dir}")

    # read in
    all_texts = []
    file_paths_to_use = []

    file_pbar = tqdm(
        total=len(absolute_filepaths),
        desc="Reading files",
        unit="files",
        position=0,
        leave=True,
        disable=not verbose
    )
    
    for filepath in absolute_filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            if text.strip():  # non-empty files
                all_texts.append(text)
                file_paths_to_use.append(filepath)
        except Exception as e:
            if verbose:
                print(f"\nError reading file {filepath.name}: {e}", file=sys.stderr)
        finally:
            file_pbar.update(1)
    file_pbar.close()

    if not all_texts:
        print("No valid text content found to analyze")
        return
    
    if verbose:
        print(f"Successfully read {len(all_texts)} files")

    # batches for processing
    batched_texts = split_batches(all_texts, batch_size=batch_size)
    batched_filepaths = split_batches(text_filepaths, batch_size=batch_size)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    batch_pbar = tqdm(
        total=len(batched_texts),
        desc="Processing batches",
        unit="batch",
        position=0,
        leave=True
    )

    for batch_idx, (texts, paths) in enumerate(zip(batched_texts, batched_filepaths)):
        try:
            if verbose:
                print(f"\nProcessing batch {batch_idx+1}/{len(batched_texts)} with {len(texts)} files")
            
            # custom version of to_tokens_and_logprobs with better error handling
            try:
                output = modified_to_tokens_and_logprobs(model, tokenizer, texts, verbose)
            except Exception as e:
                print(f"Error in token processing: {e}")
                batch_pbar.update(1)
                continue
            
            # try catch for csvs
            for fp, df in zip(paths, output):
                try:
                    output_fp = output_root / f"{fp.stem}.csv"
                    df.to_csv(output_fp, index=False)
                except Exception as e:
                    if verbose:
                        print(f"Error writing file {fp.name}: {e}")
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}: {e}")
        finally:
            batch_pbar.update(1)
    # Close batch processing progress bar
    batch_pbar.close()
            
    if verbose:
        print(f"Surprisal calculation complete. Results saved to {output_dir}")
    return

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