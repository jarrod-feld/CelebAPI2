import os
import requests
from typing import Dict, Any

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test.png")
API_URL = "http://localhost:8000/analyze"
NUM_RESULTS = 8

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

        print(f"Reading image from {TEST_IMAGE_PATH}")
        with open(TEST_IMAGE_PATH, 'rb') as f:
            # Create the multipart form data
            files = {
                "file": ("test.png", f, "image/png")
            }
            params = {
                "num_results": NUM_RESULTS
            }
            
            print(f"Sending request to {API_URL}")
            print(f"Parameters: {params}")
            
            response = requests.post(
                API_URL,
                files=files,
                params=params,  # Use params instead of data for query parameters
                timeout=30  # Add timeout
            )
            
            response.raise_for_status()  # Raise exception for bad status codes
            
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
