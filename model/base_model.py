from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_base_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
