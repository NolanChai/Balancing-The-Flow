import torch
from pprint import pprint
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
from pathlib import Path

from utils import to_tokens_and_logprobs

TEXT_FILES_ROOTDIR = "../Generations"
OUTPUT_ROOTDIR = "../Surprisal_outputs"
MODEL_NAME = "gpt2"

def calc_surprisal(model, tokenizer, input_dir, output_dir, num_files=-1):
    # Setting up files
    root = Path(input_dir)
    text_filepaths = list(root.iterdir())
    absolute_filepaths = [file.resolve() for file in text_filepaths]

    # Reading in files
    texts=[]
    for filepath in tqdm(absolute_filepaths[:num_files]):
        with open(filepath, 'r') as file:
            text = file.read()
        texts.append(text)

    # calculating surprisals with model
    batch = to_tokens_and_logprobs(model, tokenizer, texts)

    # writing ouput files
    output_root = Path(output_dir)
    for fp, text in zip(text_filepaths, batch):
        output_fp = output_root / fp.with_suffix(".csv").name
        text.to_csv(output_fp)

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, padding=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    calc_surprisal(model, tokenizer, TEXT_FILES_ROOTDIR, OUTPUT_ROOTDIR, num_files=20)