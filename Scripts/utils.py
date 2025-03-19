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
from urllib.parse import urljoin, urlparse
import nltk
from tqdm import tqdm

# nltk.download('punkt_tab')
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    input_ids = tokenizer(input_texts, padding="max_length", truncation=True, return_tensors="pt").input_ids#.to(device)
    print("Running through model:")
    t0 = time.time()
    outputs = model(input_ids)
    t1 = time.time()
    print(f"Total time: {t1 - t0}")
    print("Calculating surprisals")
    probs = torch.softmax(outputs.logits, dim=-1).detach()
    # probs = torch.softmax(outputs.logits, dim=-1).cpu().detach()
    surprisals = -1 * np.log2(probs)

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    # input_ids.cpu().detach()
    surprisals = surprisals[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_surprisals = torch.gather(surprisals, 2, input_ids[:, :, None]).squeeze(-1)

    # gather all the surprisals for the sequences into a neat table
    print("Creating surprisal tables:")
    batch = []
    sentence_id = 0
    for i, id_surp in enumerate(zip(input_ids, gen_surprisals)):
        sentence = []
        input_sentence, input_surprisals = id_surp
        print(f"Processing document {i}")
        for token, p in tqdm(zip(input_sentence, input_surprisals)):
            if token not in tokenizer.all_special_ids:
                sentence.append({
                    # "sentence_id": sentence_id,
                    "token": tokenizer.decode(token),
                    "surprisal": p.item()
                })
        batch.append(pd.DataFrame(sentence))
    return batch