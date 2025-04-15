from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up a text-generation pipeline
model_path = "D:/LLMProjects2025/GPT2/gpt_v3"  # Path to your fine-tuned model checkpoint directory
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Generate text based on a prompt
prompt = "Provide a comparison of the verse John 20:17"
output = generator(prompt, max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])
