import json
import base64
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage, db
import numpy as np
import tempfile
from PIL import Image
import os

# Initialize Firebase
if not firebase_admin._apps:
    # Load Firebase service account credentials from environment variable
    encoded_credentials = os.environ.get('GOOGLE_CREDENTIALS')
    if encoded_credentials:
        decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
        cred = credentials.Certificate(json.loads(decoded_credentials))  # Load JSON from decoded string
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'pictora-7f0ad.appspot.com',
            'databaseURL': 'https://pictora-7f0ad-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
    else:
        raise ValueError("Firebase credentials are not set in the environment variables.")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_faces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at path {image_path}")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cropped_faces = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]

    return cropped_faces

def get_embedding(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    if encodings:
        return encodings[0]
    return None

def verify_and_store_profile_embeddings(event_code):
    manual_ref = db.reference(f'rooms/{event_code}/manualParticipants')
    participants_ref = db.reference(f'rooms/{event_code}/participants')

    manual_data = manual_ref.get() or {}
    joined_data = participants_ref.get() or {}

    participants_data = {**manual_data, **joined_data}

    if not participants_data:
        return {"status": "error", "message": f"No participant data found for event {event_code}."}

    verified_embeddings = {}
    for participant_id, data in participants_data.items():
        profile_url = data.get("photoUrl")
        if profile_url:
            try:
                path_start = profile_url.find("/o/") + 3
                path_end = profile_url.find("?alt=media")
                relative_path = profile_url[path_start:path_end].replace("%2F", "/")

                blob = storage.bucket().blob(relative_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    blob.download_to_filename(temp_file.name)
                    faces = crop_faces(temp_file.name)
                    if faces:
                        embedding = get_embedding(faces[0])
                        if embedding is not None:
                            verified_embeddings[participant_id] = embedding
            except Exception as e:
                return {"status": "error", "message": f"Error processing participant {participant_id}: {str(e)}"}

    np.save(f"{event_code}_verified_embeddings.npy", verified_embeddings)
    return {"status": "success", "message": f"Saved verified embeddings for {len(verified_embeddings)} participants."}

def identify_uploaded_photos(event_code):
    embeddings_file = f"{event_code}_verified_embeddings.npy"
    if not os.path.exists(embeddings_file):
        return {"status": "error", "message": "Verified embeddings file not found. Please verify and store embeddings first."}
    
    verified_embeddings = np.load(embeddings_file, allow_pickle=True).item()

    room_ref = db.reference(f'rooms/{event_code}')
    room_data = room_ref.get()

    host_id = room_data.get("hostId")
    host_folder_path = room_data.get("hostUploadedPhotoFolderPath")
    unmatched_folder = f"rooms/{event_code}/unmatched/"
    unmatched_counter = 1

    if not host_folder_path:
        return {"status": "error", "message": f"No host uploaded photo folder path found for event {event_code}."}

    bucket = storage.bucket()
    for photo_blob in bucket.list_blobs(prefix=host_folder_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_uploaded_photo:
            try:
                photo_blob.download_to_filename(temp_uploaded_photo.name)

                faces = crop_faces(temp_uploaded_photo.name)
                if not faces:
                    continue

                matched_participants = set()
                for face in faces:
                    uploaded_embedding = get_embedding(face)
                    if uploaded_embedding is None:
                        continue

                    for participant_id, verified_embedding in verified_embeddings.items():
                        distance = np.linalg.norm(uploaded_embedding - verified_embedding)
                        if distance < 0.45:  # Match threshold
                            matched_participants.add(participant_id)

                if matched_participants:
                    for participant_id in matched_participants:
                        guest_folder = f"rooms/{event_code}/{participant_id}/photos/"
                        matched_blob = bucket.blob(f"{guest_folder}{os.path.basename(photo_blob.name)}")
                        matched_blob.upload_from_filename(temp_uploaded_photo.name)
                else:
                    unmatched_path = f"{unmatched_folder}unmatched_{unmatched_counter}.jpg"
                    unmatched_blob = bucket.blob(unmatched_path)
                    unmatched_blob.upload_from_filename(temp_uploaded_photo.name)
                    unmatched_counter += 1
            except Exception as e:
                return {"status": "error", "message": f"Error processing photo {photo_blob.name}: {str(e)}"}
            finally:
                os.remove(temp_uploaded_photo.name)

    return {"status": "success", "message": "Photo identification completed."}
