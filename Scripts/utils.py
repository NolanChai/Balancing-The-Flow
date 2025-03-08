import numpy as np
from bs4 import BeautifulSoup
import re 
import time
import random
import requests
import csv
import scrapy # may or may not try with this
import pandas
import json
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
    first_para = article.find('p', class_="paragraph--lite").text
    first_para = first_para.strip()
    sentences = sent_tokenize(first_para)  # Tokenizes text into sentences
    return sentences[0] if sentences else first_para