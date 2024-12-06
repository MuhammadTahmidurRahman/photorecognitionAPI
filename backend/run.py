import os
import sys

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# Import the 'create_app' function from 'app'
from app import create_app

# Create the Flask app
app = create_app()

# Vercel expects the app object to be named 'app'
# This allows it to serve the Flask app without explicitly calling app.run()
if __name__ != "__main__":
    application = app  # Assign the app to 'application' for Vercel's Python server
