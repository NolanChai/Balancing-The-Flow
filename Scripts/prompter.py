from openai import OpenAI
from pathlib import Path
from bs4 import BeautifulSoup
from utils import prompt_hfds, generate
import requests
from datasets import load_dataset

import re
import json
import logging

if __name__ == "__main__":
    HOST = "http://localhost:1234/v1"
    CLIENT = OpenAI(base_url=HOST, api_key="lm-studio")
    prompt_hfds(5, client=CLIENT, temperature=0.7, model_name="mistral-7b-v0.1")
