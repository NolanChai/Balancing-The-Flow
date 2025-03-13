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

# Code written by Dr. Alex Warstadt
def to_tokens_and_logprobs(model, tokenizer, input_texts):
    # move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    input_ids = tokenizer(input_texts, padding="max_length", truncation=True, return_tensors="pt").input_ids#.to(device)
    outputs = model(input_ids)
    probs = torch.softmax(outputs.logits, dim=-1).detach()
    # probs = torch.softmax(outputs.logits, dim=-1).cpu().detach()
    surprisals = -1 * np.log2(probs)

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    # input_ids.cpu().detach()
    surprisals = surprisals[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_surprisals = torch.gather(surprisals, 2, input_ids[:, :, None]).squeeze(-1)

    # gather all the surprisals for the sequences into a neat table
    batch = []
    sentence_id = 0
    for input_sentence, input_surprisals in zip(input_ids, gen_surprisals):
        sentence = []
        for token, p in zip(input_sentence, input_surprisals):
            if token not in tokenizer.all_special_ids:
                sentence.append({
                    # "sentence_id": sentence_id,
                    "token": tokenizer.decode(token),
                    "surprisal": p.item()
                })
        batch.append(pd.DataFrame(sentence))
    return batch