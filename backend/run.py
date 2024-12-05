import os
import sys

# Add the project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# Now import the 'create_app' function from 'app'
from app import create_app

# Create the Flask app using the 'create_app' function
app = create_app()

# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True)