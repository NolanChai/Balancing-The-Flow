# Balancing-The-Flow
Code for Balancing the Flow: An Information-Theoretic Study of RLHF-Induced Uniformity in Language Model Outputs

All code is available in the [Scripts Folder](https://github.com/NolanChai/Balancing-The-Flow/tree/main/Scripts). Alternative to running scripts, there are a few Python Jupyter notebooks to explore our results.

For base models, we're generating article completions by only providing the title and first sentence of each article. For each model, we have different configurations based on
the recommended default values given by their respective teams.

Llama 2 7B:
```
temperature=0.9
top_p=0.6
```

Mistral 7B:
```
temperature=0.7
```

Our environment and project is configured by the Astral [uv](https://github.com/astral-sh/uv) project manager, with all dependencies stored in `pyproject.toml`.
You can install the package through
```
pip install uv
```
Followed by
```
cd [scripts directory]
uv sync
```
to install all dependencies. We used the following command-line argument to run our experiments:
```
uv run prompter.py llama-2-7b@q8_0 -g 2000 verbose=True
```
If there are any missing dependencies, please let us know! You can also use `uv add [package]` for an update to the dependencies and PR.

Each 2000 generations took approximately 4 hours.

## General Use

Calculating Surprisal (analyze only will only compute surprisals without generating):
```
uv run prompter.py gpt2 --analyze-human -v
uv run prompter.py llama-2-7b-32k-instruct --analyze-only -g 300 -v
uv run prompter.py llama-2-7b@q8_0 --analyze-only -g 300 -v
uv run prompter.py mistral-7b-instruct-v0.3 --analyze-only -g 300 -v
uv run prompter.py mistral-7b-v0.1 --analyze-only -g 300 -v
```
To generate from scratch and compute surprisals:
```
uv run prompter.py gpt2 --analyze-human -v
uv run prompter.py llama-2-7b-32k-instruct -g 300 -s "Provided only the following article title and first sentence, complete the rest of the article from this moment onwards:" -v
uv run prompter.py llama-2-7b@q8_0 --analyze-only -g 300 -v
uv run prompter.py mistral-7b-instruct-v0.3 -g 300 -t 0.7 -p 0.95 -s "Provided only the following article title and first sentence, complete the rest of the article from this moment onwards:" -v
uv run prompter.py mistral-7b-v0.1 --analyze-only -g 300 -v
```
Other prompter flags:
```python
parser.add_argument('model', type=str, help='Model name to use for generation')
parser.add_argument('-g', '--generate', type=int, default=300, help='Number of examples to generate')
parser.add_argument('-t', '--temperature', type=float, default=0.9, help='Temperature for generation')
parser.add_argument('-p', '--top-p', type=float, default=1.0, help='Top-p (nucleus sampling) parameter')
parser.add_argument('-s', '--system-prompt', type=str, help='System prompt to prepend to each generation')
parser.add_argument('-r', '--regenerate', action='store_true', help='Regenerate existing outputs')
parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose information')
parser.add_argument('--max-tokens', type=int, default=2048, help='Maximum tokens for generation')
parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries for failed generations')
parser.add_argument('--analyze-human', action='store_true', help='Analyze human texts instead of generating new ones')
parser.add_argument('--human-dir', type=str, default='../Sources', help='Directory containing human texts to analyze')
parser.add_argument('--analyze-only', action='store_true', help='Only analyze surprisals without generating new texts')
```
UID Analysis
```
uv run analyze_uid.py --input-dir "../Surprisals/human_texts" --output-dir "../UID_Analysis/human_texts"
uv run analyze_uid.py --input-dir "../Surprisals/llama-2-7b-32k-instruct" --output-dir "../UID_Analysis/llama-2-7b-32k-instruct"
uv run analyze_uid.py --input-dir "../Surprisals/llama-2-7b@q8_0" --output-dir "../UID_Analysis/llama-2-7b@q8_0"
uv run analyze_uid.py --input-dir "../Surprisals/mistral-7b-instruct-v0.3" --output-dir "../UID_Analysis/mistral-7b-instruct-v0.3"
uv run analyze_uid.py --input-dir "../Surprisals/mistral-7b-v0.1" --output-dir "../UID_Analysis/mistral-7b-v0.1"
```
All comparison
```
uv run compare_uid.py --directories "../UID_Analysis/human_texts" "../UID_Analysis/llama-2-7b-32k-instruct" "../UID_Analysis/llama-2-7b@q8_0" "../UID_Analysis/mistral-7b-instruct-v0.3" "../UID_Analysis/mistral-7b-v0.1" --output-dir "../UID_Comparison/all_models"
```
Human vs all
```
uv run compare_uid.py --directories "../UID_Analysis/human_texts" "../UID_Analysis/llama-2-7b-32k-instruct" "../UID_Analysis/llama-2-7b@q8_0" "../UID_Analysis/mistral-7b-instruct-v0.3" "../UID_Analysis/mistral-7b-v0.1" --output-dir "../UID_Comparison/human_vs_all"
```
Llama models
```
uv run compare_uid.py --directories "../UID_Analysis/llama-2-7b-32k-instruct" "../UID_Analysis/llama-2-7b@q8_0" --output-dir "../UID_Comparison/llama_models"
```
Mistral models
```
uv run compare_uid.py --directories "../UID_Analysis/mistral-7b-instruct-v0.3" "../UID_Analysis/mistral-7b-v0.1" --output-dir "../UID_Comparison/mistral_models"
```
Human vs Each
```
uv run compare_uid.py --directories "../UID_Analysis/human_texts" "../UID_Analysis/llama-2-7b-32k-instruct" --output-dir "../UID_Comparison/human_vs_llama_32k"
uv run compare_uid.py --directories "../UID_Analysis/human_texts" "../UID_Analysis/llama-2-7b@q8_0" --output-dir "../UID_Comparison/human_vs_llama_q8"
uv run compare_uid.py --directories "../UID_Analysis/human_texts" "../UID_Analysis/mistral-7b-instruct-v0.3" --output-dir "../UID_Comparison/human_vs_mistral_instruct"
uv run compare_uid.py --directories "../UID_Analysis/human_texts" "../UID_Analysis/mistral-7b-v0.1" --output-dir "../UID_Comparison/human_vs_mistral_v0.1"
```
