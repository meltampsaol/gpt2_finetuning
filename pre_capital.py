from transformers import AutoTokenizer
from datasets import load_dataset

# Step 1: Load the dataset
ds = load_dataset("Lots-of-LoRAs/task1146_country_capital")
print("Dataset loaded successfully!")

# Step 2: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Step 3: Fix the padding token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Assign eos_token as pad_token

# Step 4: Define the preprocessing function
def preprocess_function(example):
    input_text = f"Country: {example['input']}"
    output_text = f"Capital: {example['output'][0]}"  # Access the first element of the output list
    
    tokenized_input = tokenizer(input_text, truncation=True, padding="max_length", max_length=128)
    tokenized_output = tokenizer(output_text, truncation=True, padding="max_length", max_length=128)
    
    # Debugging lengths
    print(f"Length of input_ids: {len(tokenized_input['input_ids'])}, Length of output_ids: {len(tokenized_output['input_ids'])}")
    
    return {
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "labels": tokenized_output["input_ids"],
    }



def preprocess_with_debugging(example):
    try:
        # Run your preprocessing function
        return preprocess_function(example)
    except Exception as e:
        print(f"Error processing example: {example}")
        raise e


tokenized_dataset = ds["train"].map(preprocess_function, batched=False)  # Disable batching temporarily for debugging

print("Dataset tokenized successfully!")

tokenized_dataset.save_to_disk("tokenized_capital_cities")
print("Tokenized dataset saved successfully!")

#print(ds["train"].to_pandas().head())
#print(tokenized_dataset[0])

