# Utilities
These are the main utilities and notes for usage across the project.

## lora.py
### Summary of LoRA (Low Rank Adaptations for Large Language Models)
[LoRA](https://arxiv.org/abs/2106.09685) represents weight matrices with two smaller matrices through low-rank decomposition. The original weight matrices are frozen, and final results combine both adapted and original weights. Fine-tuning is much more efficient this way since you have less parameters to train, and is orthogonal to many other parameter-efficient methods. Performance is comparable to regular fine-tuning, and does not add any inference latency because adapter weights can be merged with the base model. 

**Important Note: To eliminate latency, use the `merge_and_unload()` function to merge the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.**

#### _Imported Functions_
- `merge_adapter()`: Merge LoRA layers
- `unmerge_adapter()`: Unmerge LoRA layers
- `unload()`: get back the base model without the merging of the active lora modules (reset to main model without any LoRAs)
- `delete_adapter()`: delete existing adapter
- `add_weighted_adapter()`: combine multiple LoRAs into a new adapter
