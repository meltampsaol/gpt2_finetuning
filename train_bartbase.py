import json
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Load the preprocessed dataset
with open("formatted_text2text_dataset.json", "r") as f:
    raw_data = json.load(f)

# Convert the raw data into columnar format
data_dict = {
    "input_texts": [entry["input"] for entry in raw_data],
    "target_texts": [entry["output"] for entry in raw_data]
}

# Create a Hugging Face Dataset
dataset = Dataset.from_dict(data_dict)

# Load the BART tokenizer and model
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    # Ensure input_texts and target_texts are lists of strings
    assert isinstance(examples["input_texts"], list)
    assert isinstance(examples["target_texts"], list)

   
    model_inputs = tokenizer(examples["input_texts"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["target_texts"], max_length=512, truncation=True, padding="max_length", return_tensors="pt").input_ids
    model_inputs["labels"] = labels
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define a data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = TrainingArguments(
    output_dir="./bart-finetuned",
    eval_strategy="steps",
    eval_steps=50,  # Less frequent evaluation
    per_device_train_batch_size=1,  # Critical for stability
    gradient_accumulation_steps=8,  # Effective batch size 8
    num_train_epochs=20,  # Reduced total steps
    learning_rate=1e-6,  # Lower initial rate
    warmup_steps=100,
    weight_decay=0.2,  # Stronger regularization
    max_grad_norm=0.1,  # Tighter gradient clipping
    lr_scheduler_type="cosine",
    optim="adafactor",  # More stable optimizer
    fp16=False,  # Enable mixed precision
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs",
    logging_steps=1,
    
)
# Split the dataset into training and evaluation sets
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Start training
trainer.train()
trainer.save_model("./bart-finetuned")
tokenizer.save_pretrained("./bart-finetuned")
model.save_pretrained("./bart-finetuned")