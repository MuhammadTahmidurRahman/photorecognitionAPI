import base64

# Open your JSON file and encode it in base64
with open('backend/pictora-7f0ad-firebase-adminsdk-hpzf5-f730a1a51c.json', 'rb') as file:
    encoded = base64.b64encode(file.read()).decode('utf-8')

# Save the base64 string to a new file
with open('service-account-key.base64', 'w') as output_file:
    output_file.write(encoded)
