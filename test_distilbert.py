from transformers import pipeline

# Load your fine-tuned model and tokenizer
model_path = "./distilbert-finetuned"  # Path to fine-tuned model
tokenizer_path = "distilbert-base-uncased"  # Path to tokenizer

# Initialize fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model_path, tokenizer=tokenizer_path)

# Test input text
test_input = "Who is the [MASK] God?"

# Get top 5 predictions
results = fill_mask(test_input, top_k=5)

# Print predictions
print("\nTop predictions:")
for res in results:
    print(f"{res['token_str']}: {res['score']:.4f}")
