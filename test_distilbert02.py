from transformers import pipeline

# Load your fine-tuned model and tokenizer
model_path = "./distilbert-finetuned"  # Adjust the path if necessary
tokenizer_path = "distilbert-base-uncased"  # Use the same tokenizer as during fine-tuning

# Initialize fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model_path, tokenizer=tokenizer_path)

# Input text with a [MASK] token
input_text = "Based on the Bible The [MASK] of our Lord Jesus Christ, is the true God."

# Get top predictions for the masked token
results = fill_mask(input_text, top_k=5)

# Print predictions
print("\nTop predictions:")
for res in results:
    print(f"{res['token_str']}: {res['score']:.4f}")
