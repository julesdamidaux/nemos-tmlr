import json
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline
import torch
import gc

MODELS = {
    "gemma": {
        "hf_id": "google/gemma-3-4b-it",
        "max_new_tokens": 400
    },
    "llama": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "max_new_tokens": 400
    }
}
PAD_TOKEN_ID = 128001
SAMPLE_SIZE = 5000
SEED = 42
OUTPUT_FILE = "results_combined.json"

# Load only the 'train' split
dataset = load_dataset("commonsense_qa", "default", split="train")
# Shuffle and sample
dataset = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

# Utility function to create prompt
def build_prompt(q):
    opts = "\n".join(f"{lab}. {txt}"
                      for lab, txt in zip(q["choices"]["label"], q["choices"]["text"]))
    return f"Question: {q['question']}\nOptions:\n{opts}\n DO NOT include any ohter text, just the answer.\nAnswer:"

# Function to extract predicted answer
def extract_pred(text, labels):
    m = re.search(r"\b(" + "|".join(labels) + r")\b", text)
    return m.group(1) if m else None

# Evaluation loop for each model
all_results = {}
for name, spec in MODELS.items():
    
    pipe = pipeline(
        "text-generation",
        model=spec["hf_id"],
        max_new_tokens=spec["max_new_tokens"],
        truncation=True,
        return_full_text=False,
        pad_token_id=PAD_TOKEN_ID
    )
    results = []
    for q in tqdm(dataset, desc=f"Evaluating {name}"):
        prompt = build_prompt(q)
        out = pipe(prompt)[0]["generated_text"].strip()
        pred = extract_pred(out, q["choices"]["label"])
        true = q["answerKey"]
        results.append({
            "question": q["question"],
            "predicted": pred,
            "true": true,
            "correct": pred == true
        })
    all_results[name] = results

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

# Save all results to a single JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"✔ Results saved to {OUTPUT_FILE}")

