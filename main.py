import numpy as np
import os
import pandas as pd
from PIL import Image
import io
import base64
import pyarrow.parquet as pq
import pyarrow as pa
from supabase import create_client, Client
from dotenv import load_dotenv
import face_recognition

load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_BUCKET = "analysislib"

print("Supabase URL:", SUPABASE_URL)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase client created.")

TRAIN_PARQUET_FILE = "train-00000-of-00001.parquet"

def load_image(sample_img):
    if isinstance(sample_img, dict):
        if "bytes" in sample_img:
            return np.array(Image.open(io.BytesIO(sample_img["bytes"])))
        else:
            raise ValueError("Unsupported image dict format")
    elif isinstance(sample_img, str):
        return np.array(Image.open(sample_img))
    elif isinstance(sample_img, (bytes, bytearray)):
        return np.array(Image.open(io.BytesIO(sample_img)))
    elif isinstance(sample_img, Image.Image):
        return np.array(sample_img)
    elif isinstance(sample_img, np.ndarray):
        return sample_img
    else:
        raise ValueError("Unsupported image format")

def encode_image_to_base64(sample_img):
    if isinstance(sample_img, dict) and "bytes" in sample_img:
        image_bytes = sample_img["bytes"]
    elif isinstance(sample_img, str):
        with open(sample_img, "rb") as f:
            image_bytes = f.read()
    elif isinstance(sample_img, (bytes, bytearray)):
        image_bytes = sample_img
    else:
        pil_img = Image.fromarray(load_image(sample_img))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")

def process_and_insert_row(record, index):
    try:
        img_data = record["image"]
        label = record["label"]
        image = load_image(img_data)
        
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print(f"No face found in image for {label}")
            return
            
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            print(f"Could not generate encoding for {label}")
            return
            
        embedding = face_encodings[0]
        
        image_base64 = encode_image_to_base64(img_data)
        image_url = upload_image_to_supabase(image_base64, str(label))
        if not image_url:
            return
            
        row_data = {
            "name": label,
            "embedding": embedding.tolist(),
            "image_url": image_url
        }
        
        try:
            response = supabase.table("celebritydataset").insert([row_data]).execute()
            print(f"Inserted record for {label}.")
        except Exception as insert_err:
            print(f"Error inserting record for {label}: {insert_err}")
    except Exception as e:
        print(f"Error processing record {index}: {e}")

def upload_image_to_supabase(image_base64, name):
    img_bytes = base64.b64decode(image_base64)
    filename = f"{name.replace(' ', '_')}_{os.urandom(4).hex()}.png"
    upload_response = supabase.storage.from_(SUPABASE_BUCKET).upload(filename, img_bytes)
    if isinstance(upload_response, dict) and upload_response.get("error"):
        print(f"Error uploading image for {name}: {upload_response.get('error')}")
        return ""
    public_url_response = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
    return public_url_response

def save_embeddings_to_parquet(embeddings_data, output_file='celebrity_embeddings.parquet'):
    """Save embeddings and metadata to parquet file"""
    df = pd.DataFrame(embeddings_data)
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} embeddings to {output_file}")

def main():
    print("Reading training parquet file...")
    df = pd.read_parquet(TRAIN_PARQUET_FILE)
    print(f"Loaded {len(df)} records from train file.")
    
    # Store embeddings while processing
    embeddings_data = []
    
    for index, record in df.iterrows():
        # Process and insert to Supabase
        process_and_insert_row(record, index)
        
        # Store embedding data
        try:
            img_data = record["image"]
            label = record["label"]
            image = load_image(img_data)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                if face_encodings:
                    embeddings_data.append({
                        'name': label,
                        'embedding': face_encodings[0].tolist()
                    })
        except Exception as e:
            print(f"Error storing embedding for record {index}: {e}")
    
    # Save embeddings to parquet file
    save_embeddings_to_parquet(embeddings_data)
    
    print("Processing test image (test.png)...")
    test_image = np.array(Image.open("test.png"))
    face_locations = face_recognition.face_locations(test_image)
    
    if not face_locations:
        print("No face found in test.png!")
        return
        
    test_encoding = face_recognition.face_encodings(test_image, face_locations)[0]
    
    embeddings_list, names_list = [], []
    for index, record in df.iterrows():
        try:
            img_data = record["image"]
            label = record["label"]
            image = load_image(img_data)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_encodings = face_recognition.face_encodings(image, face_locations)
                if face_encodings:
                    embeddings_list.append(face_encodings[0])
                    names_list.append(label)
        except Exception as e:
            print(f"Error in test processing record {index}: {e}")
    
    distances = [1 - face_recognition.face_distance([emb], test_encoding)[0] for emb in embeddings_list]
    sorted_indices = np.argsort(distances)[::-1]  # Sort in descending order for similarity
    top3_indices = sorted_indices[:3]
    
    print("\nTop 3 closest matches (from training records):")
    for rank, idx in enumerate(top3_indices, start=1):
        similarity = distances[idx]
        print(f"{rank}. {names_list[idx]} - Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()