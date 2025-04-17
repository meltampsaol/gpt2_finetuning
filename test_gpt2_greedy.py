from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt_vct")  # Fine-tuned model directory
tokenizer = GPT2Tokenizer.from_pretrained("gpt_vct")
tokenizer.pad_token = tokenizer.eos_token

# Function for text generation using greedy search
def generate_greedy(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # Convert prompt to tokens
    output_ids = model.generate(input_ids, max_length=max_length, do_sample=False)  # Greedy search
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)  # Decode the output

# Interactive loop for user testing
print("Testing the fine-tuned GPT-2 model. Type 'quit' or 'exit' to stop.")
while True:
    user_input = input("Enter a prompt: ")  # Get input from the user
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting the testing loop. Goodbye!")
        break
    else:
        response = generate_greedy(user_input)  # Generate response using the fine-tuned model
        print("Generated Response:", response)
