import numpy as np
from bs4 import BeautifulSoup
import re 
import time
import random
import requests
import csv
from urllib.parse import urljoin, urlparse

START_URLS = [
    "https://www.bbc.com/",
    "https://edition.cnn.com/",
    "https://www.reuters.com/",
    "https://apnews.com/",
]

MAX_ARTICLES = 300 # temporary limit
OUTPUT_CSV = "articles.csv"

def is_valid(url):
    parsed = urlparse(url)
    valid_connect = bool(parsed.netloc)
    valid_scheme = bool(parsed.scheme)
    return valid_connect and valid_scheme