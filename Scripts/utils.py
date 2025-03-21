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
    data = load_dataset("abisee/cnn_dailymail", "1.0.0", trust_remote_code=True)['train']
    shuffled_data = data.shuffle()
    if num_articles < 0:
        num_articles = 10_000
    for i in tqdm(range(num_articles)):
        generation_output_path = Path(model_output_dir) / Path(f"{model_name}_{i}.txt")
        source_output_path = Path(human_output_dir) / Path(f"human_{i}.txt")
        if not regenerate and generation_output_path.exists():
            continue
        article = data[i]
        first_sent = get_first_sentence(article['article'])
        generation = generate(client=client, prompt=first_sent, temperature=temperature, model=model_name)
        with open(generation_output_path, "w") as outfile:
            outfile.write(generation)
        if not source_output_path.exists() and write_source:
            with open(source_output_path, "w") as outfile:
                outfile.write(article['article'])


def generate(client, prompt, temperature, model):
  completion = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "user", "content": prompt}
    ],
    temperature=temperature
  )
  return completion.choices[0].message.content

def split_batches(lst, batch_size):
    num_batches = int(np.ceil(len(lst) / batch_size))
    batched = []
    for i in range(num_batches):
        start_idx = i * batch_size
        batch = lst[start_idx:start_idx + batch_size]
        batched.append(batch)
    return batched

def calc_surprisal(model, tokenizer, input_dir, output_dir, num_files=-1, batch_size=20):
    # Setting up files
    root = Path(input_dir)
    text_filepaths = list(root.iterdir())
    absolute_filepaths = [file.resolve() for file in text_filepaths]

    # Reading in files
    if num_files > 0:
        num_files = len(absolute_filepaths + 1)
    all_texts=[]
    for filepath in tqdm(absolute_filepaths[:num_files]):
        with open(filepath, 'r') as file:
            text = file.read()
        all_texts.append(text)

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