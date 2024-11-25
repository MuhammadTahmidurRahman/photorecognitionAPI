# Import necessary modules
import sys
import firebase_admin
from firebase_admin import credentials, storage, db
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import tempfile
import os
from facenet_pytorch import MTCNN  # For face cropping
import hashlib

# Ensure model_resnet.py is in the path
sys.path.append('/content/model_resnet.py')  # Add /content to the Python path
from model_resnet import ResNet_152  # Import ResNet_152 from model_resnet

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("/content/pictora-7f0ad-firebase-adminsdk-hpzf5-58ee3b6c20.json")  # Update path
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'pictora-7f0ad.appspot.com',
        'databaseURL': 'https://pictora-7f0ad-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

print("Firebase initialized successfully.")

# Model and preprocessing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet_152((112, 112)).to(device).eval()  # Load IR-152 model

mtcnn = MTCNN(image_size=112, margin=0, post_process=False, device=device)  # For face cropping

# Transformation to normalize tensors for ResNet
transform_to_rgb = transforms.Compose([
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Convert grayscale to RGB
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize as per ResNet requirements
])

# Function to normalize embeddings
def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

# Crop multiple faces using MTCNN
def crop_faces(image_path, debug=False):
    try:
        img = Image.open(image_path).convert("RGB")
        faces = mtcnn(img, return_prob=False)  # Detect multiple faces
        if faces is not None and len(faces.shape) == 4:  # Multiple faces detected
            print(f"Detected {faces.shape[0]} faces in {image_path}.")
            processed_faces = []
            for idx, face in enumerate(faces):
                face = transform_to_rgb(face)  # Ensure RGB format
                if debug:
                    # Save cropped faces for debugging
                    face_img = transforms.ToPILImage()(face)
                    face_img.save(f"debug_cropped_face_{idx}.jpg")
                processed_faces.append(face)
            return processed_faces
        elif faces is not None:  # Single face detected
            print(f"Detected 1 face in {image_path}.")
            face = transform_to_rgb(faces)
            if debug:
                # Save cropped face for debugging
                face_img = transforms.ToPILImage()(face)
                face_img.save("debug_cropped_face.jpg")
            return [face]
        else:
            print(f"No faces detected in {image_path}.")
    except Exception as e:
        print(f"Error cropping faces: {e}")
    return []  # Return an empty list if no faces are detected

# Get embeddings for an image
def get_embedding(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        embedding = model(image_tensor)
    return normalize_embedding(embedding.cpu().numpy())

# Step 1: Verify Profile Photos and Generate Embeddings
def verify_and_store_profile_embeddings(event_code):
    participants_ref = db.reference(f'rooms/{event_code}/manualParticipants')
    participants_data = participants_ref.get()

    if not participants_data:
        print(f"No participant data found for event {event_code}.")
        return

    embeddings = {}
    for participant_id, data in participants_data.items():
        profile_url = data.get("photoUrl")
        if profile_url:
            print(f"Processing profile photo for participant: {participant_id}")
            try:
                path_start = profile_url.find("/o/") + 3
                path_end = profile_url.find("?alt=media")
                relative_path = profile_url[path_start:path_end].replace("%2F", "/")

                # Download the photo as a temporary file
                blob = storage.bucket().blob(relative_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    blob.download_to_filename(temp_file.name)

                    # Crop face and compute embedding
                    faces = crop_faces(temp_file.name, debug=True)
                    if len(faces) > 0:  # If faces are detected
                        embedding = get_embedding(faces[0])  # Use the first detected face
                        embeddings[participant_id] = embedding
                        print(f"Stored embedding for participant: {participant_id}")
            except Exception as e:
                print(f"Error processing participant {participant_id}: {e}")

    # Save verified embeddings to a NumPy file
    np.save(f"{event_code}_verified_embeddings.npy", embeddings)
    print(f"Saved verified embeddings for {len(embeddings)} participants.")

# Step 2: Identify Uploaded Photos Based on Verified Embeddings
def identify_uploaded_photos(event_code):
    # Load verified embeddings
    embeddings_file = f"{event_code}_verified_embeddings.npy"
    if not os.path.exists(embeddings_file):
        print("Verified embeddings file not found. Please verify and store embeddings first.")
        return
    verified_embeddings = np.load(embeddings_file, allow_pickle=True).item()

    # Get host-uploaded photos
    uploaded_photos_ref = db.reference(f'rooms/{event_code}/hostUploadedPhotoFolderPath')
    host_uploaded_folder_path = uploaded_photos_ref.get()
    if not host_uploaded_folder_path:
        print(f"No host uploaded photo folder path found for event {event_code}.")
        return

    print(f"Processing uploaded photos in: {host_uploaded_folder_path}")

    bucket = storage.bucket()
    unmatched_folder = f"rooms/{event_code}/unmatched/"
    unmatched_counter = 1  # Counter for unmatched photos

    for photo_blob in bucket.list_blobs(prefix=host_uploaded_folder_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_uploaded_photo:
            photo_blob.download_to_filename(temp_uploaded_photo.name)

            # Crop all faces and generate embeddings
            faces = crop_faces(temp_uploaded_photo.name, debug=True)
            if len(faces) == 0:  # If no faces are detected
                print(f"No faces detected in {photo_blob.name}. Skipping.")
                continue

            matched_participants = set()
            for face_tensor in faces:
                uploaded_embedding = get_embedding(face_tensor)  # Generate normalized embedding

                # Compare with verified embeddings
                for participant_id, verified_embedding in verified_embeddings.items():
                    distance = np.linalg.norm(uploaded_embedding - verified_embedding)
                    print(f"Distance to {participant_id}: {distance:.4f}")  # Debugging distances
                    if distance < 0.45:  # Match threshold
                        matched_participants.add(participant_id)
                        print(f"Matched face with participant {participant_id} (Distance: {distance:.4f})")

            if matched_participants:
                for participant_id in matched_participants:
                    guest_folder = f"rooms/{event_code}/{participant_id}/photos/"
                    matched_blob = bucket.blob(f"{guest_folder}{os.path.basename(photo_blob.name)}")
                    matched_blob.upload_from_filename(temp_uploaded_photo.name)
                print(f"Photo matched with participants: {', '.join(matched_participants)}")
            else:
                unmatched_path = f"{unmatched_folder}unmatched_{unmatched_counter}.jpg"
                unmatched_blob = bucket.blob(unmatched_path)
                unmatched_blob.upload_from_filename(temp_uploaded_photo.name)
                unmatched_counter += 1
                print(f"Unmatched photo saved for review: {unmatched_path}")

        os.remove(temp_uploaded_photo.name)

    print("Photo identification completed.")

# Run the pipeline
event_code = "EMC7P9XK"  # Replace with actual event code
verify_and_store_profile_embeddings(event_code)
identify_uploaded_photos(event_code)