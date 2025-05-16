from utils import *
from utils import LLM

test = LLM.Model(model="llama-2-7b")
test.batch_generate(n=5)

from utils.uid_metrics import UID

uid = UID("gpt2")
metrics = uid.run_pipeline(
    input_dir="/Users/nolan/Documents/GitHub/Balancing-The-Flow/Generations",
    metrics_file="gpt2_metrics.csv",
    plot_file="gpt2_metrics.png"
)