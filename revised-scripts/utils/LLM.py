from openai import OpenAI # type: ignore
import datetime
import io
import os
import csv
import sys
import time
import functools
import traceback
from pathlib import Path
from utils.spinner import Spinner
from utils.utils import log, log_time
from utils.utils import get_dataset
from tqdm.auto import tqdm  # Use auto for better detection
from tqdm import tqdm # type: ignore
from datasets import load_dataset # type: ignore
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

"""
An LLM Class for OpenAI/OpenAI-like APIs
"""
class Model():
    def __init__(self, model, host_url="http://localhost:1234/v1", api_key="lm-studio", verbose=True):
        self.model = model
        self.host_url = host_url
        self.api_key = api_key
        self.verbose = verbose
        self.client = OpenAI(base_url=host_url, api_key=api_key)
        print(f"{log_time()} Successfully connected to API at {self.host_url}.")
    
    def generate(self, prompt, temperature=0.9, max_tokens=2048, top_p=0.6, system_prompt=None, use_spinner=True):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if use_spinner:
            with Spinner(message=f"{log_time()} Generating response", spinner_type="braille") as spinner:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    response = completion.choices[0].message.content
                    spinner.stop(success=True)
                    return response
                except Exception as e:
                    spinner.stop(success=False)
                    log(f"ERROR: {str(e)}")
                    return None
        else:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                return completion.choices[0].message.content
            except Exception as e:
                log(f"ERROR: {str(e)}")
                return None
    def batch_generate(self, n=100, routine="article", dataset="abisee/cnn_dailymail", 
                      out=None, temperature=0.7, max_tokens=2048, top_p=0.9,
                      system_prompt=None, split='test', seed=42, max_retries=3,
                      cache_dir="datasets_cache"):
        if out is None:
            out = f"outputs/{self.model}/{routine}"
        
        output_dir = Path(out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name_short = dataset.split("/")[-1] if "/" in dataset else dataset
        csv_file = output_dir / f"{dataset_name_short}_{routine}_{n}.csv"
        
        try:
            shuffled_data, from_cache = get_dataset(
                dataset_name=dataset, 
                split=split, 
                cache_dir=cache_dir, 
                seed=seed
            )
        except Exception as e:
            log(f"ERROR: Failed to load dataset {dataset}: {str(e)}")
            return None
        
        n = min(n, len(shuffled_data))
        fieldnames = ['id', 'prompt', 'completion', 'source', 'status', 'time_taken']
        
        last_id = -1
        if csv_file.exists():
            try:
                with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_id = max(int(row['id']) for row in rows if row['id'].isdigit())
                log(f"Resuming from ID {last_id+1}")
            except Exception as e:
                log(f"ERROR reading existing CSV, starting from beginning: {str(e)}")
                last_id = -1
        
        file_exists = csv_file.exists() and last_id >= 0
        
        try:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40, complete_style="green", finished_style="green"),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                MofNCompleteColumn(),
                "•",
                TimeElapsedColumn(),
                "<",
                TimeRemainingColumn(),
                expand=True
            ) as progress:
                task = progress.add_task(f"[cyan]Generating with {self.model}", total=n)
                
                if last_id >= 0:
                    progress.update(task, completed=last_id + 1)
                
                generated = 0
                retry_count = 0
                error_count = 0
                
                def get_routine_prompt(item, routine_type):
                    if routine_type == "article":
                        article = item.get('article', '')
                        if not article:
                            return None
                        
                        title = item.get('highlights', '').split('.')[0].strip()
                        sentences = article.split('.')
                        first_sentence = sentences[0].strip() if sentences else ""
                        
                        if not first_sentence:
                            return None
                        
                        return f"Write a news article based on this title and beginning:\n\nTitle: {title}\n\nBeginning: {first_sentence}."
                        
                    elif routine_type == "story":
                        prompt_text = item.get('prompt', '')
                        if not prompt_text:
                            return None
                        
                        return f"Write a creative story based on this writing prompt:\n\n{prompt_text}"
                        
                    elif routine_type == "dialog":
                        if 'dialog' not in item:
                            return None
                            
                        dialog = item['dialog']
                        if not dialog or len(dialog) < 3:
                            return None
                        
                        dialog_sample = "\n".join([f"Person {i%2 + 1}: {turn}" for i, turn in enumerate(dialog[:3])])
                        return f"Continue this dialog naturally:\n\n{dialog_sample}"
                        
                    else:
                        log(f"Unknown routine: {routine_type}")
                        return None
                
                with open(csv_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    start_idx = last_id + 1
                    
                    for i in range(start_idx, n):
                        idx = i % len(shuffled_data)
                        item = shuffled_data[idx]
                        
                        prompt = get_routine_prompt(item, routine)
                        if not prompt:
                            continue
                        
                        progress.update(task, description=f"[cyan]Generating {i+1}/{n}")
                        
                        start_time = time.time()
                        completion = None
                        status = "success"
                        
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": prompt})
                        
                        for retry in range(max_retries):
                            try:
                                if retry > 0:
                                    progress.update(task, description=f"[yellow]Retry {retry}/{max_retries} for {i+1}/{n}")
                                
                                completion = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    top_p=top_p
                                ).choices[0].message.content
                                
                                if completion and len(completion.strip()) > 0:
                                    if retry > 0:
                                        progress.update(task, description=f"[green]Retry succeeded {i+1}/{n}")
                                    else:
                                        progress.update(task, description=f"[cyan]Generating {i+1}/{n}")
                                    break
                                
                                retry_count += 1
                                time.sleep(1)
                            except Exception as e:
                                retry_count += 1
                                time.sleep(1)
                                progress.update(task, description=f"[red]Error on try {retry+1}/{max_retries}")
                                
                                if retry == max_retries - 1:
                                    completion = f"[ERROR] {str(e)}"
                                    status = "error"
                                    error_count += 1
                        
                        time_taken = round(time.time() - start_time, 2)
                        
                        source = ""
                        if routine == "article" and "article" in item:
                            source = item["article"]
                        elif routine == "story" and "story" in item:
                            source = item["story"]
                        elif routine == "dialog" and "dialog" in item:
                            source = "\n".join(item["dialog"])
                        
                        writer.writerow({
                            'id': i,
                            'prompt': prompt,
                            'completion': completion,
                            'source': source,
                            'status': status,
                            'time_taken': time_taken
                        })
                        
                        f.flush()
                        
                        if i % 5 == 0:
                            progress.update(task, description=f"[blue]Gen: {generated+1}, Retry: {retry_count}, Err: {error_count}")
                        
                        progress.update(task, advance=1)
                        generated += 1
                
                log(f"Batch generation complete: {generated} generated, {retry_count} retries, {error_count} errors")
                return None
        
        except ImportError:
            log("Rich library not found. Using standard progress tracking.")
            
            with tqdm(
                total=n,
                desc=f"Generating with {self.model}",
                unit="examples",
                leave=True
            ) as pbar:
                
                if last_id >= 0:
                    pbar.update(last_id + 1)
                
                generated = 0
                retry_count = 0
                error_count = 0
                
                def get_routine_prompt(item, routine_type):
                    if routine_type == "article":
                        article = item.get('article', '')
                        if not article:
                            return None
                        
                        title = item.get('highlights', '').split('.')[0].strip()
                        sentences = article.split('.')
                        first_sentence = sentences[0].strip() if sentences else ""
                        
                        if not first_sentence:
                            return None
                        
                        return f"Write a news article based on this title and beginning:\n\nTitle: {title}\n\nBeginning: {first_sentence}."
                        
                    elif routine_type == "story":
                        prompt_text = item.get('prompt', '')
                        if not prompt_text:
                            return None
                        
                        return f"Write a creative story based on this writing prompt:\n\n{prompt_text}"
                        
                    elif routine_type == "dialog":
                        if 'dialog' not in item:
                            return None
                            
                        dialog = item['dialog']
                        if not dialog or len(dialog) < 3:
                            return None
                        
                        dialog_sample = "\n".join([f"Person {i%2 + 1}: {turn}" for i, turn in enumerate(dialog[:3])])
                        return f"Continue this dialog naturally:\n\n{dialog_sample}"
                        
                    else:
                        log(f"Unknown routine: {routine_type}")
                        return None
                
                with open(csv_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    start_idx = last_id + 1
                    
                    for i in range(start_idx, n):
                        idx = i % len(shuffled_data)
                        item = shuffled_data[idx]
                        
                        prompt = get_routine_prompt(item, routine)
                        if not prompt:
                            continue
                        
                        start_time = time.time()
                        completion = None
                        status = "success"
                        
                        for retry in range(max_retries):
                            try:
                                messages = []
                                if system_prompt:
                                    messages.append({"role": "system", "content": system_prompt})
                                messages.append({"role": "user", "content": prompt})
                                
                                completion = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=messages,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    top_p=top_p
                                ).choices[0].message.content
                                
                                if completion and len(completion.strip()) > 0:
                                    break
                                
                                retry_count += 1
                                time.sleep(1)
                            except Exception as e:
                                retry_count += 1
                                time.sleep(1)
                                if retry == max_retries - 1:
                                    completion = f"[ERROR] {str(e)}"
                                    status = "error"
                                    error_count += 1
                        
                        time_taken = round(time.time() - start_time, 2)
                        
                        source = ""
                        if routine == "article" and "article" in item:
                            source = item["article"]
                        elif routine == "story" and "story" in item:
                            source = item["story"]
                        elif routine == "dialog" and "dialog" in item:
                            source = "\n".join(item["dialog"])
                        
                        writer.writerow({
                            'id': i,
                            'prompt': prompt,
                            'completion': completion,
                            'source': source,
                            'status': status,
                            'time_taken': time_taken
                        })
                        
                        f.flush()
                        pbar.update(1)
                        generated += 1
                        
                        if i % 5 == 0:
                            pbar.set_description(f"Gen: {generated}, Retry: {retry_count}, Err: {error_count}")
                
                log(f"Batch generation complete: {generated} generated, {retry_count} retries, {error_count} errors")
                return None