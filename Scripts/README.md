# :cyclone: Scripts
This is our collection of scripts used to scrape and gather data for our study. Currently, we only use data taken from the [Daily Mail](https://www.dailymail.co.uk/ushome/index.html)
site due to our limited scope as a class project to test for examples. 

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

After generating on the test articles, we run tests on the RLHF'd instruct models by using the following system prompt:
```
Provided the following article title and first sentence, generate the rest of the article:
```
