Calculating Surprisal (analyze only will only compute surprisals without generating):
```
uv run prompter.py gpt2 --analyze-human -v
uv run prompter.py llama-2-7b-32k-instruct --analyze-only -g 300 -v
uv run prompter.py llama-2-7b@q8_0 --analyze-only -g 300 -v
uv run prompter.py mistral-7b-instruct-v0.3 --analyze-only -g 300 -v
uv run prompter.py mistral-7b-v0.1 --analyze-only -g 300 -v
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