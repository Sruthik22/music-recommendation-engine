import numpy as np
import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MusicDataLoader:
    """Load real music data with actual embeddings from Deezer"""
    
    def __init__(self, data_file: str = "processed_tracks.json"):
        self.data_file = data_file
        self.tracks_cache = None
        self.mood_centroids_cache = None
    
    def load_tracks(self) -> Dict:
        """Load tracks from processed data file"""
        if self.tracks_cache is not None:
            return self.tracks_cache
            
        if not os.path.exists(self.data_file):
            logger.error(f"Data file {self.data_file} not found! Please run data_pipeline.py first.")
            return {}
        
        try:
            with open(self.data_file, 'r') as f:
                track_list = json.load(f)
            
            # Convert list to dict format expected by the app
            tracks_dict = {}
            for i, track in enumerate(track_list):
                track_id = track.get("id", f"track_{i:03d}")
                
                # Map Deezer fields to our expected format
                processed_track = {
                    "id": track_id,
                    "title": track.get("title", "Unknown Title"),
                    "artist": track.get("artist", "Unknown Artist"),
                    "album": track.get("album", "Unknown Album"),
                    "tags": track.get("genres", ["unknown"]),  # Use genres as tags
                    "genres": track.get("genres", ["unknown"]),
                    "duration_ms": track.get("duration_ms", 240000),
                    "popularity": self._normalize_popularity(track.get("popularity", 0)),
                    "audio_embedding": track.get("audio_embedding", []),
                    "text_embedding": track.get("text_embedding", []),
                    "deezer_url": track.get("deezer_url", ""),
                    "preview_url": track.get("preview_url", ""),
                    "explicit": track.get("explicit", False),
                    "artist_id": track.get("artist_id"),
                    "album_id": track.get("album_id")
                }
                
                # Validate embeddings
                if not processed_track["audio_embedding"] or not processed_track["text_embedding"]:
                    logger.warning(f"Track {track_id} missing embeddings, skipping")
                    continue
                
                tracks_dict[track_id] = processed_track
            
            logger.info(f"Loaded {len(tracks_dict)} real tracks from {self.data_file}")
            self.tracks_cache = tracks_dict
            return tracks_dict
            
        except Exception as e:
            logger.error(f"Failed to load real tracks: {e}")
            return {}
    
    def _normalize_popularity(self, deezer_rank: int) -> int:
        """Convert Deezer rank to 0-100 popularity score"""
        if deezer_rank > 500000:
            return 90
        elif deezer_rank > 200000:
            return 70
        elif deezer_rank > 50000:
            return 50
        elif deezer_rank > 10000:
            return 30
        else:
            return 10
    
    def compute_genre_centroids(self, tracks: Dict) -> Dict:
        """Compute genre centroids from real track data"""
        if self.mood_centroids_cache is not None:
            return self.mood_centroids_cache
        
        genre_centroids = {}
        
        # Group tracks by genre
        tracks_by_genre = {}
        for track in tracks.values():
            genres = track.get("genres", ["unknown"])
            for genre in genres:
                if genre not in tracks_by_genre:
                    tracks_by_genre[genre] = []
                tracks_by_genre[genre].append(track)
        
        # Compute centroid for each genre that has at least 1 track
        for genre, genre_tracks in tracks_by_genre.items():
            if len(genre_tracks) >= 1:  # Use single track as centroid if needed
                # Compute centroids from genre tracks
                audio_embeddings = np.array([track["audio_embedding"] for track in genre_tracks])
                text_embeddings = np.array([track["text_embedding"] for track in genre_tracks])
                
                audio_centroid = np.mean(audio_embeddings, axis=0)
                text_centroid = np.mean(text_embeddings, axis=0)
                
                genre_centroids[genre] = {
                    "audio_centroid": audio_centroid.tolist(),
                    "text_centroid": text_centroid.tolist(),
                    "description": self._get_genre_description(genre),
                    "num_tracks": len(genre_tracks),
                    "sample_tracks": [track["title"] for track in genre_tracks[:3]]
                }
                
                logger.info(f"Computed {genre} centroid from {len(genre_tracks)} tracks")
            else:
                logger.warning(f"Genre '{genre}' has {len(genre_tracks)} track(s), skipping centroid")
        
        self.mood_centroids_cache = genre_centroids
        return genre_centroids
    
    def _get_genre_description(self, genre: str) -> str:
        """Get description for music genre"""
        descriptions = {
            "rock": "Electric guitars, drums, and powerful vocals",
            "pop": "Catchy melodies and mainstream appeal",
            "hip-hop": "Rhythmic rap vocals over strong beats",
            "electronic": "Synthesized sounds and digital production",
            "jazz": "Improvisation, complex harmonies, and swing rhythms",
            "classical": "Orchestral compositions and formal structures",
            "indie": "Independent, alternative sound with creative freedom",
            "folk": "Traditional acoustic instruments and storytelling",
            "r&b": "Soulful vocals with rhythm and blues influences",
            "country": "Rural themes with guitars and harmonica",
            "reggae": "Jamaican rhythms with off-beat emphasis",
            "blues": "Emotional expression through guitar and vocals",
            "ambient": "Atmospheric soundscapes for background listening",
            "dance": "Upbeat rhythms designed for dancing",
            "alternative": "Non-mainstream rock with experimental elements",
            "metal": "Heavy guitars with aggressive and intense sound"
        }
        return descriptions.get(genre.lower(), f"{genre.capitalize()} music")
    
    def _get_genre_energy_level(self, genre: str) -> str:
        """Get energy level category for genre"""
        energy_levels = {
            # Calm/Relaxing
            "ambient": "calm",
            "jazz": "calm", 
            "classical": "calm",
            "folk": "calm",
            "blues": "moderate",
            
            # Moderate Energy
            "indie": "moderate",
            "alternative": "moderate", 
            "r&b": "moderate",
            "reggae": "moderate",
            "country": "moderate",
            "hip-hop": "moderate",
            
            # High Energy
            "rock": "energetic",
            "metal": "energetic",
            "electronic": "energetic", 
            "dance": "energetic",
            "pop": "energetic"
        }
        return energy_levels.get(genre.lower(), "moderate")
    
    def _get_genre_activities(self, genre: str) -> List[str]:
        """Get suitable activities for genre"""
        activities = {
            "ambient": ["study", "focus", "meditation", "background", "relaxation"],
            "jazz": ["study", "dinner", "relaxation", "sophistication", "conversation"],
            "classical": ["study", "concentration", "sophistication", "formal events"],
            "folk": ["relaxation", "storytelling", "acoustic sessions", "camping"],
            "blues": ["emotional expression", "late night", "introspection"],
            "indie": ["creative work", "artistic projects", "coffee shops"],
            "alternative": ["driving", "creative work", "artistic expression"],
            "r&b": ["romantic", "smooth listening", "emotional connection"],
            "reggae": ["relaxation", "beach", "laid-back vibes", "positive energy"],
            "country": ["road trips", "rural settings", "storytelling", "nostalgia"],
            "hip-hop": ["urban lifestyle", "confidence building", "modern culture"],
            "rock": ["driving", "energy boost", "motivation", "rebellion"],
            "metal": ["workout", "intense focus", "high energy", "aggression release"],
            "electronic": ["parties", "clubs", "energy", "dancing", "modern vibes"],
            "dance": ["dancing", "parties", "celebration", "high energy", "clubs"],
            "pop": ["mainstream appeal", "singing along", "upbeat mood", "social"]
        }
        return activities.get(genre.lower(), ["general listening"])

def load_tracks() -> Dict:
    """Load tracks from processed data"""
    loader = MusicDataLoader()
    return loader.load_tracks()

def load_genre_centroids() -> Dict:
    """Load genre centroids from real data"""
    loader = MusicDataLoader()
    tracks = loader.load_tracks()
    
    if not tracks:
        logger.error("No tracks available to compute genre centroids")
        return {}
    
    return loader.compute_genre_centroids(tracks)

# Analysis functions for real data
def analyze_dataset(data_file: str = "processed_tracks.json") -> Dict:
    """Analyze the collected dataset"""
    if not os.path.exists(data_file):
        return {"error": "Dataset file not found"}
    
    loader = MusicDataLoader(data_file)
    tracks = loader.load_tracks()
    
    if not tracks:
        return {"error": "No tracks loaded"}
    
    # Basic statistics
    total_tracks = len(tracks)
    genres = {}
    duration_values = []
    popularity_values = []
    explicit_count = 0
    
    for track in tracks.values():
        # Count genres
        for genre in track.get("genres", ["unknown"]):
            genres[genre] = genres.get(genre, 0) + 1
        
        # Collect numeric values that we actually have
        if track.get("popularity"):
            popularity_values.append(track["popularity"])
        if track.get("duration_ms"):
            duration_values.append(track["duration_ms"] / 1000)  # Convert to seconds
        if track.get("explicit"):
            explicit_count += 1
    
    # Compute statistics
    analysis = {
        "total_tracks": total_tracks,
        "genres": dict(sorted(genres.items(), key=lambda x: x[1], reverse=True)),
        "duration_stats": {
            "mean_seconds": np.mean(duration_values) if duration_values else 0,
            "std_seconds": np.std(duration_values) if duration_values else 0,
            "min_seconds": min(duration_values) if duration_values else 0,
            "max_seconds": max(duration_values) if duration_values else 0
        } if duration_values else None,
        "popularity_stats": {
            "mean": np.mean(popularity_values) if popularity_values else 0,
            "std": np.std(popularity_values) if popularity_values else 0,
            "min": min(popularity_values) if popularity_values else 0,
            "max": max(popularity_values) if popularity_values else 0
        } if popularity_values else None,
        "explicit_tracks": explicit_count,
        "explicit_percentage": (explicit_count / total_tracks * 100) if total_tracks > 0 else 0,
        "embedding_stats": {
            "audio_embedding_dim": len(list(tracks.values())[0]["audio_embedding"]) if tracks else 0,
            "text_embedding_dim": len(list(tracks.values())[0]["text_embedding"]) if tracks else 0
        }
    }
    
    return analysis

def export_dataset_summary(data_file: str = "processed_tracks.json", 
                          output_file: str = "dataset_summary.json") -> bool:
    """Export dataset analysis to file"""
    try:
        analysis = analyze_dataset(data_file)
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Dataset summary exported to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to export dataset summary: {e}")
        return False
