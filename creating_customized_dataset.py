from datasets import Dataset

# ✅ A list of conversation turns (user + assistant alternating)
custom_data = [
    {"text": "What's the capital of France?", "role": "prompter"},
    {"text": "The capital of France is Paris.", "role": "assistant"},
    
    {"text": "Can you give me a fun fact about it?", "role": "prompter"},
    {"text": "Sure! Paris was originally a Roman city called Lutetia.", "role": "assistant"},
    
    {"text": "Who is the true God?", "role": "prompter"},
    {"text": "According to the Bible, the Father of our Lord Jesus Christ is the true God.", "role": "assistant"},
    
    {"text": "Can you give me a fun fact about it?", "role": "prompter"},
    {"text": "Sure! Here are some verses from the Bible: John 17:1,3 and John 20:17.", "role": "assistant"},
    
    {"text": "Is Christ also God?", "role": "prompter"},
    {"text": "No. Our Lord Jesus Christ is the Son of God (Matthew 3:17), and He never claimed to be God.", "role": "assistant"},
    
    {"text": "So, what is the origin of the belief that Christ is God?", "role": "prompter"},
    {"text": (
        "The belief that Christ is God originated at the Council of Nicaea in 325 AD (see *The Doctrine of God*, p. 59).\n"
        "The Council of Nicaea was convened by Emperor Constantine I to address the Arian controversy, which questioned "
        "Christ's divine nature and His relationship to God the Father. The council ultimately affirmed the divinity of Christ, "
        "leading to the Nicene Creed, which states that Jesus is 'true God from true God.'\n\n"
        "However, this decision was politically motivated, as Constantine aimed to unify the Christian church under a single doctrine.\n\n"
        "This is the origin of the belief that Christ is God, which is not supported by the Bible. The Bible clearly teaches that "
        "Jesus is the Son of God, not God Himself.\n\n"
        "Supporting Bible verses:\n"
        "1. John 17:1,3 — Jesus refers to the Father as the only true God.\n"
        "2. John 20:17 — Jesus tells Mary Magdalene that He is ascending to 'my Father and your Father, to my God and your God.'"
    ), "role": "assistant"},
]


# ✅ Load your list of dicts into a Dataset
custom_dataset = Dataset.from_list(custom_data)

# ✅ Preprocess function to format the conversation text
def preprocess_function(examples):
    texts = []
    for text, role in zip(examples["text"], examples["role"]):
        if role == "prompter":
            texts.append(f"<user>: {text}")
        elif role == "assistant":
            texts.append(f"<assistant>: {text}")
    return {"text": texts}

# ✅ Apply the preprocessing function to the dataset
tokenized_custom = custom_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=10,
    remove_columns=custom_dataset.column_names
)

# ✅ Display the formatted results
for i, sample in enumerate(tokenized_custom.select(range(len(tokenized_custom)))):
    print(f"Sample {i + 1}:")
    print(sample)
    print()
