from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset('text', data_files={'train': 'bible_verses.txt'})

# Load and update the tokenizer with a padding token
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Alternatively, use add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    outputs["labels"] = outputs["input_ids"].copy()  # Set labels for loss computation
    return outputs

tokenized_dataset = dataset['train'].map(tokenize_function, batched=True)
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt_v3",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1,
    max_grad_norm=0.2,
    fp16=False,
    no_cuda=True,
    report_to="none",
    save_strategy="steps",
    save_steps=5,
    save_total_limit=1,
    warmup_steps=10,
    load_best_model_at_end=False,
    logging_first_step=True,
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
trainer.save_model("./gpt_v3")
tokenizer.save_pretrained("./gpt_v3")