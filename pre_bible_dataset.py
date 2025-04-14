import json

# Load the JSON file
with open('bibleverses.json', 'r') as json_file:
    data = json.load(json_file)

# Open a text file for writing
with open('bible_verses.txt', 'w') as txt_file:
    for entry in data['data']:  # Assuming your JSON has a "data" key containing the verses
        txt_file.write(f"Verse: {entry['verse']}\n")
        for translation in entry['translations']:
            txt_file.write(f"Translation ({translation['label']}): {translation['text']}\n")
        txt_file.write("\n===\n\n")

print("Conversion completed! Saved as bible_verses.txt.")
