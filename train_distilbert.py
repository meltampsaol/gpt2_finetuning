from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForMaskedLM, Trainer, TrainingArguments

# Load your custom local dataset
dataset = load_dataset("json", data_files="./my_dataset/data.jsonl", split="train")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]})


model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()
trainer.save_model("./distilbert-finetuned")
tokenizer.save_pretrained("./distilbert-finetuned")