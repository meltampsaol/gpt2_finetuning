import json
import random
from transformers import AutoTokenizer

# Load the dataset
raw_data = [
   {"text": "What's the capital of France?", "role": "prompter"},
   {"text": "The capital of France is Paris.", "role": "assistant"},
   {"text": "What's the capital of the Philippines?", "role": "prompter"},
   {"text": "The capital of the Philippines is Manila.", "role": "assistant"},
   {"text": "What's the capital of Zamboanga del Sur?", "role": "prompter"},
   {"text": "The capital of Zamboanga del Sur is Pagadian City.", "role": "assistant"},
   {"text": "Can you give me a fun fact about it?", "role": "prompter"},
   {"text": "Sure! Paris was originally a Roman city called Lutetia.", "role": "assistant"},
   {"text": "Who is the true God?", "role": "prompter"},
   {"text": "Based on the Bible The Father of our Lord Jesus Christ.", "role": "assistant"},
   {"text": "Can you give me a fun fact about it?", "role": "prompter"},
   {"text": "Sure! I can give you verses from the Bible: John 17:1,3, John 20:17.", "role": "assistant"},
   {"text": "Is Christ also God?", "role": "prompter"},
   {"text": "No, Our Lord Jesus Christ, is the Son of God(Mat 3:17), and He never taught that He is", "role": "assistant"},
   {"text": "So, what is the origin of Christ-God teaching or belief?", "role": "prompter"},
   {"text": "It was originated in the Council of Nicaea in 325 AD (The Doctrine of God, p. 59):\nThe Council of Nicaea was convened by Emperor Constantine I in 325 AD to address the Arian controversy, which questioned the nature of Christ and his relationship to God the Father. The council ultimately affirmed the divinity of Christ, leading to the Nicene Creed, which states that Jesus is 'true God from true God.' This decision was politically motivated, as Constantine sought to unify the Christian church under a single doctrine.\nThis is the origin of the belief that Christ is God, which is not supported by the Bible. The Bible clearly states that Jesus is the Son of God and not God Himself.\nThe Bible verses that support this are:\n1. John 17:1,3 - Jesus refers to the Father as the only true God.\n2. John 20:17 - Jesus tells Mary Magdalene to tell the disciples that he is ascending to 'my Father and your Father, to my God and your God.'", "role": "assistant"},
]

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess function to create masked language modeling data
def preprocess_for_mlm(raw_data):
    tokenized_data = []
    current_prompt = None

    for entry in raw_data:
        if entry["role"] == "prompter":
            current_prompt = entry["text"]
        elif entry["role"] == "assistant" and current_prompt:
            full_text = f"{current_prompt} {entry['text']}"
            inputs = tokenizer(full_text, truncation=True, max_length=128)
            
            # Randomly mask tokens in the assistant's part
            mask_indices = random.sample(range(len(inputs["input_ids"])), k=int(0.15 * len(inputs["input_ids"])))
            for idx in mask_indices:
                if inputs["input_ids"][idx] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                    inputs["input_ids"][idx] = tokenizer.mask_token_id

            tokenized_data.append(inputs)
            current_prompt = None

    return tokenized_data

# Call the preprocessing function
tokenized_data = preprocess_for_mlm(raw_data)

json_serializable_data = []
for item in tokenized_data:
    json_serializable_data.append({
        "input_ids": item["input_ids"],  # Already a list
        "attention_mask": item["attention_mask"]  # Already a list
    })

# Save to JSON file
with open("mlm_preprocessed_data.json", "w") as f:
    json.dump(json_serializable_data, f, indent=4)

