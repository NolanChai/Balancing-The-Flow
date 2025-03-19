from openai import OpenAI
from pathlib import Path
from bs4 import BeautifulSoup
from utils import *
import requests
from datasets import load_dataset

import re
import json
import logging

# # Point to the local server
# HOST = "http://localhost:1234/v1"
# CLIENT = OpenAI(base_url=HOST, api_key="lm-studio")
# # prompt = "The following is an 800-word news article with the title \"Scientists aiming to bring back woolly mammoth create woolly mice\": A plan to revive the mammoth is on track, scientists have said after creating a new species: the woolly mouse."
# # prompt = "Here's a funny joke."
# PROMPT=""
# ARTICLE_URLS = ["https://lite.cnn.com/2025/03/06/politics/newsom-trans-athletes-womens-sports/index.html"]
# PATH = Path("../Generations/test_output")
# MODEL = "model-identifier"

# for i, url in enumerate(ARTICLE_URLS):
#   if not is_valid(url):
#     continue
#   response = requests.get(url)
#   soup = BeautifulSoup(response.text, 'lxml')
#   title = get_title(soup)
#   first_sentence = get_first_sentence(soup)
#   print(title)
#   print(first_sentence)

#   # Engineer prompt
#   suggested_article_length = 500 # TODO: replace this with num. words in article

#   prompt = f"The following is a {suggested_article_length}-word article with the title \"{title}\": {first_sentence}"

#   completion = CLIENT.chat.completions.create(
#     model="model-identifier",
#     messages=[
#       {"role": "user", "content": prompt}
#     ],
#     temperature=0.7,
#   )
  
#   out_path = Path("../Generations/test_output")
#   with open(out_path, "w") as file:
#     file.write(completion.choices[0].message.content)

# def generate(host, client, article_urls, path, prompt):
#   for i, url in enumerate(article_urls):
#     if not is_valid(url):
#       continue
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'lxml')
#     title = get_title(soup)
#     first_sentence = get_first_sentence(soup)
#     # print(title)
#     # print(first_sentence)

#     PROMPT = f"The following is a {suggested_article_length}-word article with the title \"{title}\": {first_sentence}"

#     # Engineer prompt
#     suggested_article_length = 500 # TODO: replace this with num. words in article

#     completion = client.chat.completions.create(
#       model=model,
#       messages=[
#         {"role": "user", "content": PROMPT}
#       ],
#       temperature=0.7,
#     )
    
#     out_path = path
#     with open(out_path, "w") as file:
#       file.write(completion.choices[0].message.content)

# generate(HOST, CLIENT, ARTICLE_URLS, PATH, PROMPT)

# def generate(host, client, prompt, temperature):
#   completion = client.chat.completions.create(
#     model=model,
#     messages=[
#       {"role": "user", "content": PROMPT}
#     ],
#     temperature=temperature
#   )
#   return completion.choices[0].message.content
