from model.base_model import load_base_model
from lora.fine_tune import apply_lora, fine_tune
from utils.data_loader import get_tokenizer, load_legal_dataset, prepare_dataset

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = load_base_model(model_name)
tokenizer = get_tokenizer(model_name)
model = apply_lora(model)

dataset = load_legal_dataset()
tokenized_dataset = prepare_dataset(dataset, tokenizer)

fine_tune(model, tokenizer, tokenized_dataset)
