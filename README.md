# 🤖 Legal LoRA Fine-Tuning with TinyLlama

Fine-tuning [TinyLlama 1.1B Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) using LoRA (Low-Rank Adaptation) for **Legal Question Answering** in a QA format.

---

## 📌 Project Highlights

- 🧠 **Model**: TinyLlama-1.1B-Chat
- ⚙️ **Tuning Method**: Parameter-efficient LoRA with PEFT
- 📄 **Dataset**: Custom legal QA pairs (JSONL format)
- 🧪 **Inference**: Legal-style answers based on formatted prompts
- 🧱 **Modular**: Clean architecture with `model`, `lora`, `utils`, `inference`

---

## 🗂️ Directory Overview
legal-lora-finetune/
├── model/ # Loads base model (TinyLlama)
├── lora/ # LoRA configuration + training
├── utils/ # Tokenization + dataset prep
├── inference/ # Inference pipeline
├── output/ # Trained adapter checkpoints
├── qa_cleaned_strict.jsonl # Cleaned QA dataset
├── run.py # Main training script
├── README.md
└── requirements.txt


---


