import requests
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset # type: ignore
from sklearn.model_selection import train_test_split

# ========== CONFIG ========== 
VERSIONS = ["nkjv", "niv"]  # You can expand this
VERSES = ["John 3:16", "Genesis 1:1", "Psalms 23:1"]
API_URL = "http://localhost:3001/dataset"

# ========== FETCH + FORMAT ========== 
def get_translations(verse, versions):
    versions_str = ",".join(versions)
    response = requests.get(f"{API_URL}?bibles={versions_str}&verse={verse}")
    return response.json() if response.status_code == 200 else None

def format_translation_sample(verse_data):
    verse = verse_data["Verse"]
    translations = verse_data["Translations"]
    
    output = f"Verse: {verse}\n"
    for version, translation in translations.items():
        output += f"{verse} {version.upper()}: {translation.strip()}\n"
    return output.strip()

# ========== PREPARE TRAINING DATA ========== 
def prepare_training_samples(verses, versions):
    samples = []
    for verse in verses:
        verse_data = get_translations(verse, versions)
        if verse_data:
            formatted = format_translation_sample(verse_data)
            samples.append(formatted)
        else:
            print(f"⚠️ Failed to fetch: {verse}")
    return samples

samples = prepare_training_samples(VERSES, VERSIONS)

# ========== TOKENIZATION ========== 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Avoids pad-token warnings

# Tokenize samples
tokenized = tokenizer(samples, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Split data into training and validation sets
train_samples, val_samples = train_test_split(samples, test_size=0.2)

# Tokenize training and validation data separately
train_tokenized = tokenizer(train_samples, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
val_tokenized = tokenizer(val_samples, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Create datasets from tokenized data
train_dataset = Dataset.from_dict({
    "input_ids": train_tokenized["input_ids"],
    "attention_mask": train_tokenized["attention_mask"],
    "labels": train_tokenized["input_ids"],  # Labels are the same for language models
})

val_dataset = Dataset.from_dict({
    "input_ids": val_tokenized["input_ids"],
    "attention_mask": val_tokenized["attention_mask"],
    "labels": val_tokenized["input_ids"],
})

# ========== MODEL ========== 
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ========== TRAINING ========== 
training_args = TrainingArguments(
    output_dir="./biblegpt",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    warmup_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",
    logging_dir="./logs",
    logging_steps=1,
    evaluation_strategy="epoch",  # Evaluate after every epoch
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Add validation data
    data_collator=data_collator,
)

# ========== GO! ========== 
trainer.train()

# ========== SAVE ========== 
trainer.save_model("./biblegpt")
tokenizer.save_pretrained("./biblegpt")
print("✅ Training complete and model saved.")
