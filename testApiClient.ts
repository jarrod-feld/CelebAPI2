import requests

TEST_IMAGE_PATH = r"c:\Users\Jarrod\Desktop\VS code porjects\celebrity face comparer - Copy\test.jpg"
API_URL = "http://localhost:8000/analyze"
USER_ID = "test-user-123"

def upload_celebrity_analysis(analysis, user_id):
    # Simulate uploading analysis result to Supabase
    return {"status": "success", "user_id": user_id}

def test_api():
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': f}
            data = {'num_results': '8'}
            print("Sending request to API...")
            response = requests.post(API_URL, files=files, data=data)
        if response.status_code != 200:
            raise Exception(f"API call failed with status {response.status_code}")
        analysis_result = response.json()
        print("API analysis result:", analysis_result)

        print("Uploading analysis result to Supabase...")
        upload_response = upload_celebrity_analysis(analysis_result, USER_ID)
        print("Supabase upload response:", upload_response)

    except Exception as error:
        print("Error during API test:", error)

if __name__ == "__main__":
    test_api()
