from transformers import AutoTokenizer
import json
import numpy as np
from prepare_datasets import prepare_daic_woz

# ---- Tokenizer loading
tokenizer_name = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


# ---- Utils
def get_dataset_length_stats(tokenizer, dataset):
    result = {}

    for split in dataset:
        lengths = []
        for element in dataset[split]:
            tokens = tokenizer(element["dialogue"], return_tensors="pt")
            lengths.append(tokens["input_ids"].shape[1])

        result[split] = {
            "max": int(np.max(lengths)),
            "min": int(np.min(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
        }

    return result


# ---- dataset loading/prep
dw_dataset = prepare_daic_woz("./data/DAIC-WOZ")
print(dw_dataset)

# ---- Stats
print("-" * 20, "Stats Overall", "-" * 20)
length_stats = get_dataset_length_stats(tokenizer, dw_dataset)
print(json.dumps(length_stats, indent=4))
