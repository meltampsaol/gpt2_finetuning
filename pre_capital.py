from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load the dataset
dataset = load_dataset("Lots-of-LoRAs/task1146_country_capital")
print("Dataset loaded successfully!")

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("Dataset tokenized successfully!")

# Save the tokenized dataset for later use
tokenized_dataset.save_to_disk("tokenized_country_capital")
print("Tokenized dataset saved successfully!")
