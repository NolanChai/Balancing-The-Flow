import os
import glob
import pandas as pd

# --- HUGGING FACE & PEFT ---
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

def create_text_from_csvs(csv_pattern):
    """
    1. Finds all CSVs matching the pattern (e.g., 'human_*.csv').
    2. Loads each CSV (with columns: 'token', 'surprisal').
    3. Concatenates the 'token' column into a text string (ignoring 'surprisal').
    Returns a Hugging Face Dataset with a single column 'text'.
    """
    # Gather all CSV paths
    filepaths = glob.glob(csv_pattern)
    if not filepaths:
        raise ValueError(f"No CSV files found for pattern: {csv_pattern}")

    all_texts = []
    for fp in filepaths:
        df = pd.read_csv(fp)
        # Convert tokens to a single string for this file:
        text_str = " ".join(df["token"].astype(str).tolist())
        all_texts.append(text_str)

    # Create a single combined text
    # (Alternatively, you might keep them separate, one example per file.)
    combined_text = "\n".join(all_texts)

    # Wrap it into a Dataset object
    dataset = Dataset.from_dict({"text": [combined_text]})
    return dataset

def main():
    # --------------------
    # 1. CREATE THE DATASET
    # --------------------
    csv_pattern = "/Users/nolan/Documents/GitHub/Balancing-The-Flow/Surprisals/human_texts/human_*.csv"
    dataset = create_text_from_csvs(csv_pattern)
    
    # Example: If you want multiple examples, you could split the combined text
    # into chunks or lines. This example only has 1 row of text.

    # --------------------
    # 2. TOKENIZE THE DATA
    # --------------------
    # We'll use a LLaMA-2 7B checkpoint from Hugging Face (requires acceptance).
    # If you're using local weights, point to your local folder.
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a pad token (LLaMA sometimes doesn't).
    # We'll map pad_token to eos_token to avoid typical pad token issues.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    # We apply the tokenize function to each row (though we currently have only 1).
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Hugging Face Trainer expects columns "input_ids" and "attention_mask".
    # The above tokenize function will produce these.

    # --------------------
    # 3. SET UP LoRA
    # --------------------
    # Basic LoRA config for a causal language model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # LLaMA-2 attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # --------------------
    # 4. LOAD THE BASE MODEL & APPLY LoRA
    # --------------------
    # For 7B LLaMA-2, you may need to load in 8-bit for GPU memory reasons.
    # Requires bitsandbytes.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )

    # Wrap with LoRA
    model = get_peft_model(model, lora_config)

    # --------------------
    # 5. TRAINING ARGUMENTS & TRAINER
    # --------------------
    training_args = TrainingArguments(
        output_dir="lora-llama2-output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,         # Increase for real training
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=25,
        save_total_limit=2,
        fp16=True,                  # Mixed precision
        push_to_hub=False          # Set True if you want to push to HF Hub
    )

    # Since we only have a single dataset row for demonstration,
    # we're just reusing the same data for eval. In practice, separate train/eval.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    # --------------------
    # 6. TRAIN
    # --------------------
    print("Starting training...")
    trainer.train()

    # Save final model (LoRA adapter weights)
    trainer.save_model("lora-llama2-output")

if __name__ == "__main__":
    main()
