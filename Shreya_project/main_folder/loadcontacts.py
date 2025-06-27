import json
import os

file_path = "contacts.json"

if os.path.exists(file_path):
    with open(file_path, "r") as file:
        try:
            contacts = json.load(file)
            print("Contacts loaded successfully.")
        except json.JSONDecodeError:
            print("Invalid JSON format.")
else:
    print("contacts.json file not found.")