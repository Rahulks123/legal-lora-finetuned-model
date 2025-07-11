import json
from datasets import Dataset
from transformers import AutoTokenizer

def get_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_legal_dataset(path="/content/qa_cleaned_strict.jsonl"):
    with open(path, "r") as f:
        lines = [json.loads(line) for line in f]
    return Dataset.from_list(lines)

def prepare_dataset(dataset, tokenizer):
    def tokenize(example):
        text = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized
