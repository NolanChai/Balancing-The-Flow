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
from utils import *


START_URLS = [
    "https://lite.cnn.com"
]

MAX_ARTICLES = 300 # temporary limit
OUTPUT_CSV = "articles.csv"

# PSEUDOCODE
# for each url we have, locate subdomains and articles
# 

articles = []

for url in START_URLS:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    # print(response.status_code) # 200 = valid
    # print(response.text)
    # print(soup.find('title').get_text())
    for subdir in soup.find_all('a')[:-5]:
        articles.append(url + subdir.get('href'))
    article_dumps = json.dumps(articles)

out = open("articles.json", "w")
out.write(article_dumps)
out.close()

# fetch all articles, go to next page
