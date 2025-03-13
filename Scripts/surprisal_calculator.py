import torch
from pprint import pprint
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import pandas as pd

import os
from pathlib import Path

from utils import to_tokens_and_logprobs

TEXT_FILES_ROOTDIR = "../Generations"
OUTPUT_ROOTDIR = "../Surprisal_outputs"

root = Path(TEXT_FILES_ROOTDIR)
text_filepaths = list(root.iterdir())
absolute_filepaths = [file.resolve() for file in text_filepaths]
texts=[]
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding=True)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

for filepath in absolute_filepaths[:10]:
    with open(filepath, 'r') as file:
        text = file.read()
    texts.append(text)

batch = to_tokens_and_logprobs(model, tokenizer, texts)
output_root = Path(OUTPUT_ROOTDIR)
for fp, text in zip(text_filepaths, batch):
    output_fp = output_root / fp.with_suffix(".csv").name
    text.to_csv(output_fp)