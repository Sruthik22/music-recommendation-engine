#!/usr/bin/env python3
"""
Demo script to showcase the music recommendation engine functionality
"""

from sentence_transformers import SentenceTransformer
from models_real import load_genre_centroids, load_tracks
from recommendation_engine import RecommendationEngine
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_demo():
    """Run a demonstration of the recommendation engine"""
    
    print("🎵 Music Recommendation Engine Demo")
    print("=" * 50)
    
    # Initialize components
    print("Loading models and data...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load real data
    try:
        tracks_db = load_tracks()
        genre_centroids = load_genre_centroids()
        
        if not tracks_db:
            print("❌ No tracks found! Please run data_pipeline.py first to collect data.")
            quit()
            
        print("✅ Using real music data")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        print("Please run data_pipeline.py first to collect data.")
        quit()
    
    # Create recommendation engine
    engine = RecommendationEngine(sentence_model, genre_centroids, tracks_db)
    
    print(f"✅ Loaded {len(tracks_db)} tracks")
    print(f"✅ Loaded {len(genre_centroids)} genre categories")
    print()
    
    # Demo requests
    demo_requests = [
        "I want chill music for studying",
        "Give me energetic workout tracks",
        "Something sad and emotional",
        "Upbeat party music",
        "Ambient background music for meditation"
    ]
    
    for i, description in enumerate(demo_requests, 1):
        print(f"Demo {i}: '{description}'")
        print("-" * 40)
        
        try:
            recommendations = engine.recommend_tracks(description, max_tracks=3)
            
            print(f"Detected genre: {recommendations['genre_detected']}")
            print(f"Genre confidence: {recommendations['genre_scores']}")
            print("Top recommendations:")
            
            for j, track_rec in enumerate(recommendations['tracks'], 1):
                track = track_rec['track']
                score = track_rec['similarity_score']
                explanation = track_rec['explanation']
                
                print(f"  {j}. {track['title']} by {track['artist']}")
                print(f"     Tags: {', '.join(track['tags'])}")
                print(f"     Score: {score:.3f}")
                print(f"     Why: {explanation}")
                print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    run_demo()
