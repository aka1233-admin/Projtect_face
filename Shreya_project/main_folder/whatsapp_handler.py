import pywhatkit
import json
import os
import threading

class WhatsAppHandler:
    def __init__(self, contacts_file="contacts.json"):
        self.contacts_file = contacts_file
        
    def load_contacts(self):
        """Load contacts from JSON file"""
        try:
            if not os.path.exists(self.contacts_file):
                print("Contacts file not found.")
                return None
                
            with open(self.contacts_file, "r") as file:
                contacts = json.load(file)
                return contacts
        except json.JSONDecodeError:
            print("Contacts file format error.")
            return None
        except Exception as e:
            print(f"Error loading contacts: {e}")
            return None
    
    def find_contact(self, contacts, name_part):
        """Find contact by name (case-insensitive)"""
        name_part = name_part.strip().lower()
        
        for contact in contacts:
            if contact.lower() in name_part or name_part in contact.lower():
                return contact
        return None
    
    def send_whatsapp_message(self, command, speak_callback=None):
        """Send WhatsApp message based on voice command"""
        def send_message():
            try:
                contacts = self.load_contacts()
                if not contacts:
                    if speak_callback:
                        speak_callback("Contacts file not found or invalid.")
                    return

                # Better parsing of contact name
                command_parts = command.split("to")
                if len(command_parts) < 2:
                    if speak_callback:
                        speak_callback("Please specify who to send the message to.")
                    return
                    
                name_part = command_parts[-1].strip().lower()
                
                # Find contact (case-insensitive)
                contact_name = self.find_contact(contacts, name_part)
                
                if contact_name and contact_name in contacts:
                    number = contacts[contact_name]
                    message = f"Hello {contact_name.title()}, the manager wants to see you."
                    
                    print(f"Sending message to {contact_name} at {number}")
                    pywhatkit.sendwhatmsg_instantly(number, message, wait_time=15, tab_close=True)
                    
                    if speak_callback:
                        speak_callback(f"Message sent to {contact_name}")
                else:
                    if speak_callback:
                        speak_callback("Contact not found. Please check the name.")
                    print(f"Available contacts: {list(contacts.keys())}")
                    
            except Exception as e:
                print(f"Error sending message: {e}")
                if speak_callback:
                    speak_callback("Failed to send message due to an error.")

        threading.Thread(target=send_message, daemon=True).start()
    
    def create_sample_contacts(self):
        """Create a sample contacts file"""
        sample_contacts = {
            "john": "+1234567890",
            "jane": "+0987654321",
            "bob": "+1122334455"
        }
        
        try:
            with open(self.contacts_file, "w") as file:
                json.dump(sample_contacts, file, indent=4)
            print(f"Sample contacts file created: {self.contacts_file}")
        except Exception as e:
            print(f"Error creating sample contacts file: {e}")
    
    def add_contact(self, name, number):
        """Add a new contact to the contacts file"""
        try:
            contacts = self.load_contacts()
            if contacts is None:
                contacts = {}
                
            contacts[name.lower()] = number
            
            with open(self.contacts_file, "w") as file:
                json.dump(contacts, file, indent=4)
                
            print(f"Contact {name} added successfully")
            return True
        except Exception as e:
            print(f"Error adding contact: {e}")
            return False
    
    def list_contacts(self):
        """List all contacts"""
        contacts = self.load_contacts()
        if contacts:
            print("Available contacts:")
            for name, number in contacts.items():
                print(f"  {name.title()}: {number}")
            return contacts
        else:
            print("No contacts available")
            return None