from datasets import load_dataset

# Load the raw dataset
dataset = load_dataset("OpenAssistant/oasst1")
train_ds = dataset["train"]

# Preprocess the dataset with your tokenize function
def preprocess_function(examples):
    texts = []
    for text, role in zip(examples["text"], examples["role"]):
        if role == "prompter":
            texts.append(f"<user>: {text}")
        elif role == "assistant":
            texts.append(f"<assistant>: {text}")
    return {"text": texts}

tokenized_train = train_ds.map(
    preprocess_function, 
    batched=True, 
    batch_size=10, 
    remove_columns=train_ds.column_names
)

# Display sample rows from tokenized_train
for index, example in enumerate(tokenized_train.select(range(5))):
    print(f"Sample {index + 1}:")
    print(example)
    print()
