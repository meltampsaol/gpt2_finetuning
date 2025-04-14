from transformers import AutoTokenizer, BartForConditionalGeneration, pipeline

# Load the tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Input text for testing
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
pipe = pipeline("text2text-generation", model="facebook/bart-base")
output = pipe("What is the capital of France?")
print(output)
