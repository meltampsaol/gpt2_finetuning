from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import numpy as np

# Step 1: Download the dataset
dataset = load_dataset("Lots-of-LoRAs/task1146_country_capital")
print("Dataset loaded successfully!")

# Step 2: Tokenize the texts
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("Dataset tokenized successfully!")

# Step 3: Split the dataset into smaller chunks
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    block_size = 128  # Set block size to your preference
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True)
print("Dataset split into smaller chunks successfully!")

# Step 4: Split dataset into train and evaluation sets
train_dataset, val_dataset = lm_dataset["train"], lm_dataset["test"]

# Step 5: Initialize Trainer
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="output_vct",  # Save training results to 'output_vct'
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Step 6: Perform Training
trainer.train()
print("Model training complete!")

# Step 7: Save the fine-tuned model
model.save_pretrained("gpt_vct")  # Save the model in 'gpt_vct'
tokenizer.save_pretrained("gpt_vct")
print("Fine-tuned model and tokenizer saved to 'gpt_vct' directory!")

# Step 8: Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)
