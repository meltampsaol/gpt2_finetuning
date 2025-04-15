from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import re

# Load model
model_path = "D:/LLMProjects2025/GPT2/gpt_v3"  # Path to your fine-tuned model checkpoint directory
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def parse_input(user_input):
    """
    Parses input like:
    - 'I John 2:1,NIV,TEV'
    - 'I John 2:1 NIV,TEV'
    """
    if ',' in user_input:
        # Find position of the last space before the first comma
        first_comma = user_input.index(',')
        before_comma = user_input[:first_comma]
        if ' ' in before_comma:
            last_space = before_comma.rindex(' ')
            verse = user_input[:last_space].strip()
            versions = user_input[last_space:].replace(verse, '').strip().split(',')
        else:
            raise ValueError("Invalid verse format.")
    else:
        # Fallback to rsplit on last space
        if ' ' in user_input:
            verse, version_string = user_input.rsplit(' ', 1)
            versions = version_string.split(',')
        else:
            raise ValueError("Invalid input format.")
    
    versions = [v.upper().strip() for v in versions if v.strip()]
    return verse.strip(), versions

def clean_generated_text(text):
    """
    Cleans unnecessary random strings or non-Bible-related content dynamically.
    """
    # Remove URLs (if any)
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove unwanted non-alphanumeric symbols except some punctuation
    text = re.sub(r"[^\w\s,.!?]", "", text)

    # Normalize excessive newlines
    text = re.sub(r"[\n\r]{2,}", "\n", text)

    return text

def filter_fabricated_lines(text, comparison_lines):
    """
    Filters fabricated lines and retains only valid comparisons.
    """
    allowed_texts = [line.split(":", 1)[1].strip() for line in comparison_lines]
    cleaned_lines = []

    for line in text.strip().splitlines():
        if any(allowed_text in line for allowed_text in allowed_texts):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# --- Main Loop ---
while True:
    user_input = input("Enter verse and versions (e.g., 'I John 2:1,NIV,TEV' or 'I John 2:1 NIV,TEV') (type 'exit' or 'quit' to stop): ").strip()
    
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the program. Goodbye!")
        break

    try:
        verse, versions = parse_input(user_input)
        invalid_versions = [v for v in versions if not v.isalnum()]
        if invalid_versions:
            raise ValueError(f"Invalid versions: {', '.join(invalid_versions)}")
    except ValueError as e:
        print(f"Input error: {e}")
        print("Valid formats:")
        print("- Comma-separated: 'I John 2:1,NIV,TEV'")
        print("- Space-separated: 'I John 2:1 NIV,TEV'")
        continue

    # --- API Call ---
    try:
        response = requests.get(
            "http://localhost:3001/dataset",
            params={"verse": verse, "bibles": ",".join(versions)},
            timeout=10
        )
        response.raise_for_status()
        translations = response.json().get("Translations", {})
    except Exception as e:
        print(f"API Error: {str(e)}")
        continue

    # --- Prompt Construction ---
    comparison_lines = []
    for version in versions:
        if version in translations:
            comparison_lines.append(f"{version}: {translations[version]}")
        else:
            print(f"Warning: Missing translation for {version}")

    if not comparison_lines:
        print("Error: No translations available for comparison")
        continue

    prompt = (
        f"Compare the following translations of {verse} only:\n\n"
        + "\n".join(comparison_lines)
    )

    # --- Generation ---
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    output = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=150,
    temperature=0.9,
    repetition_penalty=1.2,
    do_sample=True,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
    )



    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    cleaned_full_text = clean_generated_text(full_text)
    cleaned_full_text = filter_fabricated_lines(cleaned_full_text, comparison_lines)

    print(f"\nBible Translations for {verse}:\n{'='*60}")
    print(cleaned_full_text)
    print("="*60)
