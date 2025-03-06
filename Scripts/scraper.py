import numpy as np
from bs4 import BeautifulSoup
import re 
import time
import random
import requests
import csv
import scrapy # may or may not try with this
import pandas
from urllib.parse import urljoin, urlparse


START_URLS = [
    "https://lite.cnn.com"
]

MAX_ARTICLES = 300 # temporary limit
OUTPUT_CSV = "articles.csv"

def is_valid(url):
    parsed = urlparse(url)
    valid_connect = bool(parsed.netloc)
    valid_scheme = bool(parsed.scheme)
    return valid_connect and valid_scheme

# PSEUDOCODE
# for each url we have, locate subdomains and articles
# 

for url in START_URLS:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    # print(response.status_code) # 200 = valid
    # print(response.text)
    # print(soup.find('title').get_text())
    for link in soup.find_all('a')[:-5]:
        print(link.get('href'))

# fetch all articles, go to next page
