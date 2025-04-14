from transformers import pipeline

# Set up a text-generation pipeline
generator = pipeline('text-generation', model="./gpt_v3", tokenizer="./gpt_v3")

# Generate text based on a prompt
prompt = "Provide a comparison of the verse John 3:16 in NIV and TEV"
output = generator(prompt, max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])
