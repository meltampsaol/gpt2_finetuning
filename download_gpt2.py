from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

# Choose your model size: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model_name = 'gpt2'

# Local directory where the model will be saved (inside current working folder)
local_dir = './local_gpt2'

# Download and cache the tokenizer and model
print("Downloading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Save both locally
print(f"Saving model and tokenizer to {local_dir}...")
tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print("Done! GPT-2 model saved locally.")
