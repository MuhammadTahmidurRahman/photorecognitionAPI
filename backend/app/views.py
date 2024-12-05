from flask import Blueprint, request, jsonify

# Existing imports
from .photo_processing import verify_and_store_profile_embeddings, identify_uploaded_photos

process_photo = Blueprint('process_photo', __name__)

# Add a route for the root URL
@process_photo.route('/', methods=['GET'])
def home():
    return "Welcome to the Photo Recognition API!"

@process_photo.route('/process-photo', methods=['POST'])
def process_photo_route():
    try:
        request_json = request.get_json()
        event_code = request_json.get("event_code")

        # Call the existing functions (verify and store embeddings, identify photos)
        verification_result = verify_and_store_profile_embeddings(event_code)
        if verification_result['status'] == 'error':
            return jsonify(verification_result), 500

        identification_result = identify_uploaded_photos(event_code)
        if identification_result['status'] == 'error':
            return jsonify(identification_result), 500

        return jsonify({"status": "success", "message": "Processed successfully."}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
