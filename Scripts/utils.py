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
                write_source=True, regenerate=True):
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
    """

    # handling for missing paths
    model_output_path = Path(model_output_dir)
    human_output_path = Path(human_output_dir)

    model_output_path.mkdir(parents=True, exist_ok=True)
    if write_source:
        human_output_path.mkdir(parents=True, exist_ok=True)

    data = load_dataset("abisee/cnn_dailymail", "1.0.0", trust_remote_code=True)['train']
    shuffled_data = data.shuffle()

    # setting articles to generate
    if num_articles < 0:
        num_articles = 10_000

    # make sure it doesnt exist dataset size
    num_articles = min(num_articles, len(shuffled_data))

    for i in tqdm(range(num_articles)):
        # output paths
        generation_output_path = Path(model_output_dir) / Path(f"{model_name}_{i}.txt")
        source_output_path = Path(human_output_dir) / Path(f"human_{i}.txt")

        # skip if exists
        if not regenerate and generation_output_path.exists():
            continue

        try:
            # article and first sentence
            article = data[i]
            first_sent = get_first_sentence(article['article'])

            # generate w prompt
            generation = generate(client=client, prompt=first_sent, temperature=temperature, model=model_name)

            # write to path
            with open(generation_output_path, "w") as outfile:
                outfile.write(generation)
            
            # write source article if needed
            if (not source_output_path.exists() or regenerate) and write_source:
                with open(source_output_path, "w") as outfile:
                    outfile.write(article['article'])
        
        except Exception as e:
            print(f"Error processing article {i}: {e}")
            continue

        print(f"Successfully generated {num_articles} articles using {model_name}")
    return


def generate(client, prompt, temperature, model):
    """Generate a completion using the given API client, model and temperature parameter

    Args:
        client : API client used for model prompting
        prompt (str): prompt to give the model
        temperature (float): temperature parameter for the model
        model (str): name of the model

    Returns:
        str: generated completion
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
        {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return completion.choices[0].message.content
    completion = client.chat.completions.create(
        model=model,
        messages=[
        {"role": "user", "content": prompt}
        ],
        temperature=temperature
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

def calc_surprisal(model, tokenizer, input_dir, output_dir, num_files=-1, batch_size=20):
    """Calculate surprisal for each token of a corpus of texts

    Args:
        model (GPT2LMHeadModel): pretrained language model to use for probability calculation
        tokenizer (GPT2Tokenizer): tokenizer associated with the model
        input_dir (str/path): path to directory containing corpus
        output_dir (str/path): desired output directory for csv files
        num_files (int, optional): number of files to analyze. Passing -1 will analyze all files in the directory.
        batch_size (int, optional): batch size for surprisal calculation input. Defaults to 20.
    """
    import os
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import time
    from tqdm import tqdm
    
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
        
    print(f"Reading {len(absolute_filepaths)} files from {input_dir}")

    # read in
    all_texts = []
    file_paths_to_use = []
    
    for filepath in tqdm(absolute_filepaths):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            if text.strip():  # non-empty files
                all_texts.append(text)
                file_paths_to_use.append(filepath)
        except Exception as e:
            print(f"Error reading file {filepath.name}: {e}")
    
    if not all_texts:
        print("No valid text content found to analyze")
        return
        
    print(f"Successfully read {len(all_texts)} files")

    # batches for processing
    batched_texts = split_batches(all_texts, batch_size=batch_size)
    batched_filepaths = split_batches(text_filepaths, batch_size=batch_size)

    # calculating surprisals with model
    for texts, paths in tqdm(zip(batched_texts, batched_filepaths)):
        output = to_tokens_and_logprobs(model, tokenizer, texts)

        # writing ouput files
        output_root = Path(output_dir)
        for fp, text in zip(paths, output):
            output_fp = output_root / fp.with_suffix(".csv").name
            text.to_csv(output_fp)