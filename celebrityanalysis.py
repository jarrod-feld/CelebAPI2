import os
import io
import numpy as np
import face_recognition
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from supabase import create_client, Client

# ...existing code for environment and client configuration...
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise Exception("Missing SUPABASE_URL or SUPABASE_ANON_KEY in environment variables.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = FastAPI()

class CelebrityResult(BaseModel):
    rank: int
    name: str
    similarity: float
    image_url: str

class AnalysisResponse(BaseModel):
    results: List[CelebrityResult]

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

async def get_total_record_count():
    try:
        response = supabase.table('celebritydataset_count').select('record_count').single().execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Record count not found")
        return response.data['record_count']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching record count: {str(e)}")

async def get_celebrity_chunk(offset: int, limit: int = 1000):
    try:
        response = supabase.table("celebritydataset")\
            .select("name, embedding, image_url")\
            .range(offset, offset + limit - 1)\
            .execute()
        return response.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching celebrity data: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_face(file: UploadFile = File(...), num_results: int = 8):
    # Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are accepted.")
    
    # Read and load image using face_recognition
    contents = await file.read()
    image_stream = io.BytesIO(contents)
    image = face_recognition.load_image_file(image_stream)
    
    # Get face encodings. We assume one face per image.
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    user_encoding = encodings[0]
    
    # Fetch celebrity records in chunks of 1000
    candidates = []
    total_records = await get_total_record_count()
    offset = 0
    chunk_size = 1000
    
    while offset < total_records:
        records = await get_celebrity_chunk(offset, chunk_size)
        if not records:
            break
            
        for record in records:
            celeb_embedding = np.array(record.get("embedding"))
            if celeb_embedding.shape != user_encoding.shape:
                continue
            distance = euclidean_distance(user_encoding, celeb_embedding)
            candidates.append({ "name": record.get("name"), "image_url": record.get("image_url"), "distance": distance })
        
        offset += chunk_size

    if not candidates:
        raise HTTPException(status_code=404, detail="No matching celebrity records found.")
    
    # Normalize similarity: closest distance gets a score of 10, farthest gets 0.
    distances = [c["distance"] for c in candidates]
    min_distance = min(distances)
    max_distance = max(distances)
    
    def compute_normalized_score(distance: float) -> float:
        if max_distance == min_distance:
            return 10.0
        return 10 * (max_distance - distance) / (max_distance - min_distance)
    
    # Add score to candidates.
    for c in candidates:
        c["score"] = compute_normalized_score(c["distance"])
    
    # Sort descending by normalized score and select top results.
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    top_results = sorted_candidates[:num_results]
    
    results = []
    for index, candidate in enumerate(top_results):
        results.append(CelebrityResult(
            rank=index + 1,
            name=candidate["name"],
            similarity=round(candidate["score"], 2),
            image_url=candidate["image_url"]
        ))
    
    return AnalysisResponse(results=results)

# To run the API use:
#   For systems where 'uvicorn' is recognized:
#       uvicorn celebrityanalysis:app --reload
#   On Windows, if 'uvicorn' is not recognized, use:
#       python -m uvicorn celebrityanalysis:app --reload
