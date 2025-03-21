import torch
from pprint import pprint
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from pathlib import Path

from utils import to_tokens_and_logprobs, calc_surprisal

# TEXT_FILES_ROOTDIR = "../Generations"
TEXT_FILES_ROOTDIR = "../Sources"
OUTPUT_ROOTDIR = "../Surprisal_outputs"
MODEL_NAME = "gpt2"

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, padding=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    calc_surprisal(model, tokenizer, TEXT_FILES_ROOTDIR, OUTPUT_ROOTDIR, num_files=-1)