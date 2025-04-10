import os
import requests
from typing import Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
API_URL = "http://localhost:8000/analyze"
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test.png")
BUCKET_NAME = "analysislib"
NUM_RESULTS = 8

def upload_image_to_supabase(file_path: str) -> str:
    """Upload image to Supabase storage and return public URL"""
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    
    # Generate unique filename
    file_ext = os.path.splitext(file_path)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Read file and upload
    with open(file_path, 'rb') as f:
        file_data = f.read()
        response = supabase.storage.from_(BUCKET_NAME).upload(unique_filename, file_data)
        
        if hasattr(response, 'error') and response.error is not None:
            raise Exception(f"Upload failed: {response.error}")
    
    # Get public URL
    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)
    return public_url

def validate_response(response_data: Dict[str, Any]) -> bool:
    """Validate that the response has the expected structure."""
    if not isinstance(response_data.get('results'), list):
        return False
    for result in response_data['results']:
        if not all(key in result for key in ['rank', 'name', 'similarity', 'image_url']):
            return False
    return True

def test_api():
    try:
        if not os.path.exists(TEST_IMAGE_PATH):
            raise FileNotFoundError(f"Test image not found at {TEST_IMAGE_PATH}")

        print(f"Uploading image to Supabase storage...")
        image_url = upload_image_to_supabase(TEST_IMAGE_PATH)
        print(f"Image uploaded successfully. URL: {image_url}")
            
        print(f"Sending request to {API_URL}")
        
        # Prepare request data
        json_data = {
            "image_url": image_url,
            "num_results": NUM_RESULTS
        }
        
        # Send request to API
        response = requests.post(
            API_URL,
            json=json_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        response.raise_for_status()
        
        data = response.json()
        if not validate_response(data):
            raise ValueError("Response format is invalid")
            
        print("\nAPI Response:")
        print("-" * 50)
        for result in data['results']:
            print(f"Rank {result['rank']}: {result['name']} "
                  f"(Similarity: {result['similarity']})")
            print(f"Image URL: {result['image_url']}")
            print("-" * 50)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_api()
