from datasets import Dataset
import random
import re

# Original user-assistant data
raw_data = [
    {"text": "What's the capital of France?", "role": "prompter"},
    {"text": "The capital of France is Paris.", "role": "assistant"},
    {"text": "Can you give me a fun fact about it?", "role": "prompter"},
    {"text": "Sure! Paris was originally a Roman city called Lutetia.", "role": "assistant"},
    {"text": "Who is the true God?", "role": "prompter"},
    {"text": "Based on the Bible The Father of our Lord Jesus Christ.", "role": "assistant"},
    {"text": "Can you give me a fun fact about it?", "role": "prompter"},
    {"text": "Sure! I can give you verses from the Bible: John 17:1,3, John 20:17.", "role": "assistant"},
    {"text": "Is Christ also God?", "role": "prompter"},
    {"text": "No, Our Lord Jesus Christ, is the Son of God(Mat 3:17), and He never taught that He is", "role": "assistant"},
    {"text": "So, what is the origin of Christ-God teaching or belief?", "role": "prompter"},
    {"text": "It was originated in the Council of Nicaea in 325 AD. The Council of Nicaea affirmed the divinity of Christ, leading to the Nicene Creed. However, the Bible says Jesus is the Son of God, not God Himself (John 17:3, John 20:17).", "role": "assistant"},
]

# Create pairs of prompter and assistant messages
mlm_formatted = []
for i in range(0, len(raw_data), 2):
    if i+1 < len(raw_data):
        prompter_text = raw_data[i]['text']
        assistant_text = raw_data[i+1]['text']
        combined = f"{prompter_text} {assistant_text}"
        mlm_formatted.append({"text": combined})

# Create dataset
mlm_dataset = Dataset.from_list(mlm_formatted)
mlm_dataset.save_to_disk("./mlm_dataset")

print("Sample dataset:")
for sample in mlm_dataset.select(range(2)):
    print(sample['text'])