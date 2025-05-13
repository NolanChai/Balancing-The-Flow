import os
import subprocess
import argparse
from pathlib import Path
import sys
import platform

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def setup_paths(base_path):
    """Setup and create necessary directories."""
    base_path = Path(base_path)
    lora_path = base_path / "improved_lora_adapters"
    tokenizer_path = base_path / "improved_tokenizer"
    output_path = base_path / "merged_model"
    gguf_path = base_path / "gguf_model"
    
    # Create output directories if they don't exist
    output_path.mkdir(exist_ok=True)
    gguf_path.mkdir(exist_ok=True)
    
    return str(lora_path), str(tokenizer_path), str(output_path), str(gguf_path)

def download_and_merge_model(model_id, lora_path, output_path):
    """Download base model and apply LoRA adapters."""
    print(f"Loading LoRA config from: {lora_path}")
    # Load the PEFT configuration to determine the base model
    peft_config = PeftConfig.from_pretrained(lora_path)
    
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model from LoRA config: {base_model_name}")
    
    # Check if the provided model_id matches the base model in PEFT config
    if base_model_name != model_id:
        print(f"Warning: The base model in the PEFT config ({base_model_name}) doesn't match the provided model_id ({model_id}).")
        response = input("Continue with the provided model_id? (yes/no): ").strip().lower()
        if response != "yes":
            print(f"Using the base model from PEFT config: {base_model_name}")
            model_id = base_model_name
    
    print(f"Loading base model: {model_id}")
    try:
        # Try to load the base model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("\nThis could be due to authentication requirements. You might need to:")
        print("1. Login to Hugging Face: `huggingface-cli login`")
        print("2. Accept the model's terms of use on the Hugging Face website")
        print("3. Try again")
        sys.exit(1)
    
    print(f"Loading LoRA adapters from: {lora_path}")
    # Load and apply the LoRA adapters
    model = PeftModel.from_pretrained(model, lora_path)
    
    print("Merging LoRA adapters with base model...")
    # Merge the LoRA adapters with the base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    # Save the merged model
    merged_model.save_pretrained(output_path, safe_serialization=True)
    
    return merged_model

def save_tokenizer(model_id, tokenizer_path, output_path):
    """Save the tokenizer to the output path."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    try:
        # Try to load from the saved tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        print(f"Loading tokenizer from model_id instead: {model_id}")
        try:
            # If that fails, load from the model_id
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            print(f"Error loading tokenizer from model_id: {e}")
            print("You might need to login to Hugging Face or accept the model's terms of use.")
            sys.exit(1)
    
    print(f"Saving tokenizer to: {output_path}")
    # Save the tokenizer
    tokenizer.save_pretrained(output_path)

def find_conversion_script(llama_cpp_path):
    """Find the appropriate conversion script in the llama.cpp repository."""
    llama_cpp_path = Path(llama_cpp_path)
    # Check for possible script names in order of likelihood
    script_names = [
        "convert.py",
        "convert-hf-to-gguf.py",
        "convert_hf_to_gguf.py",
        "convert_transformers_to_gguf.py"
    ]
    
    # Also check in subdirectories
    for script_name in script_names:
        # Check in root directory
        script_path = llama_cpp_path / script_name
        if script_path.exists():
            return script_path
            
        # Check in python directory if it exists
        python_dir = llama_cpp_path / "python"
        if python_dir.exists():
            script_path = python_dir / script_name
            if script_path.exists():
                return script_path
                
        # Check in scripts directory if it exists
        scripts_dir = llama_cpp_path / "scripts"
        if scripts_dir.exists():
            script_path = scripts_dir / script_name
            if script_path.exists():
                return script_path
    
    # If we couldn't find any of the expected scripts, look for any .py files that might be the converter
    print("Could not find expected conversion script, searching for potential conversion scripts...")
    possible_scripts = []
    
    for py_file in llama_cpp_path.glob("**/*.py"):
        if "convert" in py_file.name.lower() and "gguf" in py_file.name.lower():
            possible_scripts.append(py_file)
    
    if possible_scripts:
        print("Found potential conversion scripts:")
        for i, script in enumerate(possible_scripts):
            print(f"{i+1}. {script}")
        
        while True:
            try:
                choice = int(input("Enter the number of the script to use (or 0 to exit): "))
                if choice == 0:
                    sys.exit(0)
                if 1 <= choice <= len(possible_scripts):
                    return possible_scripts[choice-1]
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    
    return None

def convert_to_gguf(llama_cpp_path, merged_model_path, gguf_output_path, model_basename, quant_type):
    """Convert the merged model to GGUF format."""
    llama_cpp_path = Path(llama_cpp_path)
    
    print("Converting model to GGUF format...")
    
    # Find the appropriate conversion script
    convert_script = find_conversion_script(llama_cpp_path)
    
    if convert_script is None:
        print("Could not find a suitable conversion script in the llama.cpp repository.")
        print("Please manually convert the model using llama.cpp tools.")
        return
    
    print(f"Using conversion script: {convert_script}")
    
    # Set output file
    output_file = Path(gguf_output_path) / f"{model_basename}-{quant_type}.gguf"
    
    # Prepare conversion command
    cmd = [
        "python", str(convert_script),
        "--outfile", str(output_file),
    ]
    
    # Add outtype parameter if it's supported
    if quant_type != "f16":
        cmd.extend(["--outtype", quant_type])
    
    # Add model path
    cmd.append(str(merged_model_path))
    
    # Run the conversion script
    print(f"Running conversion command: {' '.join(cmd)}")
    try:
        # Change to the directory containing the script to avoid import errors
        original_dir = os.getcwd()
        os.chdir(str(convert_script.parent))
        
        # Run the conversion
        subprocess.run(cmd, check=True)
        
        # Return to original directory
        os.chdir(original_dir)
        
        print(f"GGUF model saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting model to GGUF: {e}")
        print("Command failed. Checking script to suggest correct arguments...")
        
        # Try to read the script to see what arguments it expects
        try:
            with open(convert_script, 'r') as f:
                content = f.read()
                print("\nScript content snippet (first 50 lines):")
                print("\n".join(content.split("\n")[:50]))
                print("\nPlease check the script requirements and try manual conversion.")
        except Exception as read_error:
            print(f"Could not read script: {read_error}")

def main():
    parser = argparse.ArgumentParser(description="Download, merge and convert a Llama model with LoRA adapters to GGUF")
    parser.add_argument("--base_path", type=str, 
                        default="C:/Users/nchai/OneDrive/Documents/GitHub/Balancing-The-Flow/Models/LoRA", 
                        help="Base path containing the LoRA adapters")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Hugging Face model ID")
    parser.add_argument("--model_basename", type=str, default="llama-2-7b-lora-merged", 
                        help="Base name for the output GGUF file")
    parser.add_argument("--llama_cpp_dir", type=str, 
                        default="C:/Users/nchai/OneDrive/Documents/GitHub/llama-cpp", 
                        help="Directory for llama.cpp")
    parser.add_argument("--quant_type", type=str, default="q4_k_m", 
                        choices=["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k_s", "q3_k_m", 
                                 "q3_k_l", "q4_k_s", "q4_k_m", "q5_k_s", "q5_k_m", "q6_k"], 
                        help="Quantization type for GGUF conversion")
    
    args = parser.parse_args()
    
    # Setup paths
    lora_path, tokenizer_path, output_path, gguf_path = setup_paths(args.base_path)
    
    # Download and merge the model
    merged_model = download_and_merge_model(args.model_id, lora_path, output_path)
    
    # Save the tokenizer
    save_tokenizer(args.model_id, tokenizer_path, output_path)
    
    # Convert to GGUF
    convert_to_gguf(args.llama_cpp_dir, output_path, gguf_path, args.model_basename, args.quant_type)
    
    print("All done!")

if __name__ == "__main__":
    main()