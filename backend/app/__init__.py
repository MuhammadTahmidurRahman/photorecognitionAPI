from flask import Flask

def create_app():
    app = Flask(__name__)

    # Initialize routes and other configurations
    from .views import process_photo  # Import the blueprint
    app.register_blueprint(process_photo)  # Register the blueprint

    return app
