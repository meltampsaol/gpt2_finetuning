from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM


model_path = "D:/LLMProjects2025/GPT2/gpt_vct"  # Path to your fine-tuned model checkpoint directory
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# Ensure padding is properly defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Example country names
test_countries = ["United States", "India", "France", "Japan", "Brazil"]

# Format the input text for the model
formatted_inputs = [f"Country: {country}\nCapital:" for country in test_countries]
for input_text in formatted_inputs:
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    # Generate output
    outputs = model.generate(
    inputs["input_ids"],
    max_length=128,
    num_return_sequences=1,
    temperature=0.5,  # Controls randomness/creativity
    do_sample=True,  # Enables sampling for temperature to take effect
    )

    
    # Decode and print predictions
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {input_text}\nPredicted Output: {decoded_output}\n")
