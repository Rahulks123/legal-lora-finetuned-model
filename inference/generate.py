from transformers import pipeline

def generate_text(model, tokenizer, prompt="### Question:\nWhat is a contract?\n\n### Answer:\n", max_length=100):
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return gen(prompt, max_length=max_length, do_sample=True)[0]['generated_text']
