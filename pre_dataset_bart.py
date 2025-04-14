import json
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
    {
        "text": (
            "It was originated in the Council of Nicaea in 325 AD (The Doctrine of God, p. 59):\n"
            "The Council of Nicaea was convened by Emperor Constantine I in 325 AD to address the Arian controversy, "
            "which questioned the nature of Christ and his relationship to God the Father. The council ultimately affirmed "
            "the divinity of Christ, leading to the Nicene Creed, which states that Jesus is 'true God from true God.' This "
            "decision was politically motivated, as Constantine sought to unify the Christian church under a single doctrine.\n"
            "This is the origin of the belief that Christ is God, which is not supported by the Bible. The Bible clearly states "
            "that Jesus is the Son of God and not God Himself.\n"
            "The Bible verses that support this are:\n"
            "1. John 17:1,3 - Jesus refers to the Father as the only true God.\n"
            "2. John 20:17 - Jesus tells Mary Magdalene to tell the disciples that he is ascending to 'my Father and your Father, "
            "to my God and your God.'"
        ),
        "role": "assistant"
    },
]
formatted_data = []
current_prompt = None
for entry in raw_data:
    if entry["role"] == "prompter":
        current_prompt = entry["text"]
    elif entry["role"] == "assistant" and current_prompt:
        formatted_data.append({
            "input": current_prompt,
            "output": entry["text"]
        })
        current_prompt = None

# Save the formatted dataset
with open("formatted_text2text_dataset.json", "w") as f:
    json.dump(formatted_data, f, indent=4)