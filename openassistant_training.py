from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os

# ========== CHECKPOINT CONFIG ==========
checkpoint_dir = "./gpt_v2"
os.makedirs(checkpoint_dir, exist_ok=True)

# ========== LOAD MODEL ==========
model_path = "./local_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.config.loss_type = "ForCausalLMLoss"  # Fix warning

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== LOAD DATASET ==========
dataset = load_dataset("OpenAssistant/oasst1")
train_ds = dataset["train"]

# ========== PREPROCESSING ==========
def preprocess_function(examples):
    texts = []
    for text, role in zip(examples["text"], examples["role"]):
        if role == "prompter":
            texts.append(f"<user>: {text}{tokenizer.eos_token}")
        elif role == "assistant":
            texts.append(f"<assistant>: {text}{tokenizer.eos_token}")
    return tokenizer(
        texts,
        truncation=True,
        max_length=384,
        padding=False,
        add_special_tokens=True
    )

# ========== PROCESS DATASETS ==========
tokenized_train = train_ds.map(
    preprocess_function,
    batched=True,
    batch_size=500,
    remove_columns=train_ds.column_names
)

# ========== DATA COLLATOR ==========
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No masked language modeling
)

# ========== TRAINING SETUP ==========
training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=5,
    fp16=False,
    use_cpu=True,  # Explicitly disable CUDA for CPU-only training
    report_to="none",
    save_strategy="steps",
    save_steps=50,  # Save every 50 steps
    save_total_limit=1,  # Keep only the latest checkpoint
    warmup_steps=10,
    load_best_model_at_end=False,
    logging_first_step=True,
    disable_tqdm=False
)

# ========== CHECKPOINT LOGIC ==========
checkpoints = [
    d for d in os.listdir(checkpoint_dir) 
    if d.startswith("checkpoint") and os.path.isdir(os.path.join(checkpoint_dir, d))
]

if checkpoints:
    checkpoint_numbers = [int(d.split("-")[-1]) for d in checkpoints]
    latest_checkpoint = f"checkpoint-{max(checkpoint_numbers)}"
    resume_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Resuming from latest checkpoint: {resume_checkpoint}")
else:
    print("Starting training from scratch")
    resume_checkpoint = None

# ========== TRAINER ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,  # Pass the full dataset, avoid row selection
    data_collator=data_collator,
)

# ========== TRAINING ==========
try:
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
except KeyboardInterrupt:
    print("\nManual interruption - saving current state...")
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Model saved at {checkpoint_dir}")

trainer.save_model(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)
print("Training complete - final model saved!")
