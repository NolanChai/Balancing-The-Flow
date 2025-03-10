from openai import OpenAI
from pathlib import Path
from bs4 import BeautifulSoup
from utils import *
import requests
from datasets import load_dataset

import re
import json
import logging

# Point to the local server
HOST = "http://localhost:1234/v1"
CLIENT = OpenAI(base_url=HOST, api_key="lm-studio")
NUM_ARTICLES = 200
logging.basicConfig(filename="prompter.log")
logger = logging.getLogger(__name__)

# with open("articles.json", 'r') as f:
#   ARTICLE_URLS = json.loads(f.read())

# articles_enumerated = dict(enumerate(ARTICLE_URLS))
# with open("articles_enum.json", "w") as out:
#   json.dump(articles_enumerated, out)
logger.info("Loading cnn_dailymail...")
articles = load_dataset("abisee/cnn_dailymail", "1.0.0", trust_remote_code=True)["train"]

# for i, url in articles_enumerated.items():
for i in range(NUM_ARTICLES):
  logger.info(f"Prompting with article #{i}")

  # Scrape Article and retrieve title and first sentence
  # if not is_valid(url):
  #   print("url invalid")
  #   continue
  # response = requests.get(url)
  # soup = BeautifulSoup(response.text, 'lxml')
  # try:
  #   title = get_title(soup)
  #   first_sentence = get_first_sentence(soup)
  #   print(f"{title}\n{first_sentence}\n")
  # except AttributeError as e:
  #   print(f"article #{i} at {url} is missing title or first paragraph\n{e}")
  #   continue
  # except Exception as e:
  #   print(f"error with article #{i} at {url}\n{type(e)}\n{e}")
  #   continue

  
  # prompt = f"{title}\n\n {first_sentence}"
  article = articles[i]['article']
  prompt = get_first_sentence(article)
  try:
    completion = CLIENT.chat.completions.create(
      model="model-identifier",
      messages=[
        {"role": "user", "content": prompt}
      ],
      temperature=0.7,
    )
  except Exception as e:
    logger.exception(e)
    continue
    
  
  out_path = Path(f"../Generations/llama_2_7b_{i}.txt")
  with open(out_path, "w") as file:
    file.write(completion.choices[0].message.content)
