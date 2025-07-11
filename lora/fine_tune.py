from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

def apply_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    return get_peft_model(model, config)

def fine_tune(model, tokenizer, dataset, output_dir="output", resume_from_checkpoint=None):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=3e-4,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,

        save_strategy="steps",       # Save by steps
        save_steps=700,              # ✅ Save checkpoint at step 700
        save_total_limit=2,          # Keep only last 2 checkpoints

        report_to="none",            # Disable WandB/Hub logging
        push_to_hub=False
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
