from datasets import load_dataset

# Load the dataset
ds = load_dataset("Lots-of-LoRAs/task1146_country_capital")

# Check available splits
#print(ds)
# Display sample data (first few rows)
#print(ds["train"].to_pandas().head())

# Or, print the first example in raw format
#print(ds["train"][0])

# Print the first 5 examples of the train split
for i in range(5):
    print(f"Input: {ds['train'][i]['input']}")
    print(f"Output: {ds['train'][i]['output']}")
    print(f"ID: {ds['train'][i]['id']}")
    print("\n")
