import json
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


with open("mlm_preprocessed_data.json", "r") as f:
    raw_data = json.load(f)

# Transform list of dictionaries into columnar format
data_dict = {
    "input_ids": [entry["input_ids"] for entry in raw_data],
    "attention_mask": [entry["attention_mask"] for entry in raw_data]
}

# Create a Hugging Face Dataset
dataset = Dataset.from_dict(data_dict)

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./distilbert-finetuned",
    eval_strategy="steps",
    eval_steps=10,
    load_best_model_at_end=True,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=1,
    max_grad_norm=0.2,
    weight_decay=0.01,
    warmup_steps=50,
    per_device_train_batch_size=4,
    num_train_epochs=50,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
)

# Split dataset into train and eval
tokenized_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = tokenized_dataset["train"]
eval_dataset = tokenized_dataset["test"]

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Start training
trainer.train()
trainer.save_model("./distilbert-finetuned")
tokenizer.save_pretrained("./distilbert-finetuned")