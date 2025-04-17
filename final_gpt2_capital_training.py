from datasets import load_from_disk
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

# Step 1: Load the Tokenized Dataset
tokenized_dataset = load_from_disk("tokenized_country_capital")
print("Tokenized dataset loaded successfully!")

# Step 2: Split the Dataset into Smaller Chunks
def group_texts(examples):
    # Concatenate all tokenized examples into a single list for each key
    concatenated = {
        k: sum([list(item) if isinstance(item, list) else [item] for item in examples[k]], []) 
        for k in examples.keys()
    }
    total_length = len(concatenated["input_ids"])
    # Drop the small remainder if the total_length is not divisible by block_size
    block_size = 128  # Set block size to your preference
    total_length = (total_length // block_size) * block_size
    # Create blocks of fixed size
    result = {
        k: [concatenated[k][i: i + block_size] for i in range(0, total_length, block_size)]
        for k in concatenated.keys()
    }
    # Add labels to match input_ids for loss computation
    result["labels"] = result["input_ids"]
    return result




lm_dataset = tokenized_dataset.map(group_texts, batched=True)
print("Dataset split into smaller chunks successfully!")

# Step 3: Split the Dataset into Training and Validation Sets
train_dataset = lm_dataset["train"]
val_dataset = lm_dataset["test"]

# Step 4: Initialize the Trainer
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="output_vct",  # Save training results here
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="logs",  # Logging directory
    logging_steps=1,
    eval_strategy="epoch",  # Evaluate at the end of each epoch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Step 5: Perform Training
trainer.train()
print("Model training complete!")

# Step 6: Save the Fine-Tuned Model
model.save_pretrained("gpt_vct")  # Save the model here
print("Fine-tuned model saved to 'gpt_vct'.")
