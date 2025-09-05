from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Recommendation Engine", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the sentence transformer model
@app.on_event("startup")
async def startup_event():
    global sentence_model, genre_centroids, tracks_db
    logger.info("Loading sentence transformer model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load mood centroids and tracks database
    try:
        from models_real import load_genre_centroids, load_tracks
        tracks_db = load_tracks()
        genre_centroids = load_genre_centroids()
        
        if not tracks_db:
            logger.error("No tracks found! Please run data_pipeline.py first to collect data.")
            quit()
            
        logger.info("Using real music data")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.error("Please run data_pipeline.py first to collect data.")
        quit()
    logger.info("Application startup complete!")

# Pydantic models for API
class PlaylistRequest(BaseModel):
    description: str
    max_tracks: Optional[int] = 10

class Track(BaseModel):
    id: str
    title: str
    artist: str
    tags: List[str]
    bpm: Optional[int] = None
    duration_ms: Optional[int] = None
    audio_embedding: List[float]
    text_embedding: List[float]

class RecommendedTrack(BaseModel):
    track: Track
    similarity_score: float
    explanation: str

class PlaylistResponse(BaseModel):
    tracks: List[RecommendedTrack]
    mood_detected: str
    total_tracks: int

@app.get("/")
async def root():
    return {"message": "Music Recommendation Engine API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/genres")
async def get_available_genres():
    """Get list of available genre categories"""
    return {"genres": list(genre_centroids.keys())}

@app.post("/recommend", response_model=PlaylistResponse)
async def generate_playlist(request: PlaylistRequest):
    """Generate a music playlist based on user description"""
    try:
        from recommendation_engine import RecommendationEngine
        
        engine = RecommendationEngine(sentence_model, genre_centroids, tracks_db)
        recommendations = engine.recommend_tracks(request.description, request.max_tracks)
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error generating playlist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating playlist: {str(e)}")

@app.get("/tracks/sample")
async def get_sample_tracks():
    """Get a sample of available tracks"""
    sample_tracks = list(tracks_db.values())[:5]
    return {"sample_tracks": sample_tracks}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
