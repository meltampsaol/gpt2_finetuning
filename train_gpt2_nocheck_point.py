from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Tokenizer
from datasets import load_from_disk
from transformers import TrainerCallback

class PlainTextLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get("loss", "N/A")
            learning_rate = logs.get("learning_rate", "N/A")
            grad_norm = logs.get("grad_norm", "N/A")
            
            # Format and print logs
            log_message = f"Step: {step}, Loss: {loss:.4f}, Learning Rate: {learning_rate:.6f}, Grad Norm: {grad_norm}"
            print(log_message)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2",pad_token_id=tokenizer.eos_token_id)


# Load the tokenized dataset
tokenized_dataset = load_from_disk("tokenized_capital_cities")
print("Tokenized dataset loaded successfully!")

# Define training arguments without checkpoint saving
training_args = TrainingArguments(
    output_dir="output_vct/",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=1,
    logging_dir="logs",
    log_level="error",  # Suppress default Trainer logging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer to the root folder
model.save_pretrained("gpt_vct")
tokenizer.save_pretrained("gpt_vct")
print("Fine-tuned model and tokenizer saved to gpt_vct folder!")
