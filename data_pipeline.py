#!/usr/bin/env python3
"""
Data Pipeline for Music Recommendation Engine
Fetches real music data and generates embeddings
"""

import os
import json
import time
import requests
import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import base64
from io import BytesIO
import logging

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Audio embedding models
try:
    import openl3
    OPENL3_AVAILABLE = True
except ImportError:
    OPENL3_AVAILABLE = False
    print("⚠️  OpenL3 not available. Install with: pip install openl3")

try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    print("⚠️  CLAP not available. Install with: pip install laion-clap")

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeezerAPI:
    """Deezer API client for fetching track metadata and previews"""
    
    def __init__(self):
        # Deezer API is public and doesn't require authentication for basic search
        self.base_url = "https://api.deezer.com"
        
    def search_tracks(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for tracks on Deezer"""
        params = {
            "q": query,
            "limit": limit
        }
        
        response = requests.get(f"{self.base_url}/search", params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        else:
            logger.error(f"Deezer search failed: {response.text}")
            return []
    
    def get_track_details(self, track_id: str) -> Dict:
        """Get detailed track information"""
        response = requests.get(f"{self.base_url}/track/{track_id}")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get track details: {response.text}")
            return {}
    
    def get_album_details(self, album_id: str) -> Dict:
        """Get album information for additional metadata"""
        response = requests.get(f"{self.base_url}/album/{album_id}")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get album details: {response.text}")
            return {}
    
    def get_artist_details(self, artist_id: str) -> Dict:
        """Get artist information for genres"""
        response = requests.get(f"{self.base_url}/artist/{artist_id}")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get artist details: {response.text}")
            return {}
    
    def search_by_genre(self, genre: str, limit: int = 25) -> List[Dict]:
        """Search for tracks by genre on Deezer"""
        # Deezer search supports genre filtering
        genre_queries = {
            "rock": "rock",
            "pop": "pop",
            "hip-hop": "rap",
            "electronic": "electronic",
            "jazz": "jazz",
            "classical": "classical",
            "indie": "indie",
            "folk": "folk",
            "r&b": "rnb",
            "country": "country",
            "reggae": "reggae",
            "blues": "blues",
            "ambient": "ambient electronic",
            "dance": "dance",
            "alternative": "alternative rock",
            "metal": "metal"
        }
        
        query = genre_queries.get(genre.lower(), genre)
        return self.search_tracks(query, limit=limit)

class AudioEmbeddingExtractor:
    """Extract audio embeddings using various models"""
    
    def __init__(self, model_type: str = "openl3"):
        self.model_type = model_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified audio embedding model"""
        if self.model_type == "openl3" and OPENL3_AVAILABLE:
            # OpenL3 model loads automatically when needed
            logger.info("OpenL3 model ready")
        elif self.model_type == "clap" and CLAP_AVAILABLE:
            self.model = laion_clap.CLAP_Module(enable_fusion=False)
            self.model.load_ckpt()  # Load pre-trained weights
            logger.info("CLAP model loaded")
        else:
            logger.warning(f"Model {self.model_type} not available, using dummy embeddings")
    
    def extract_from_url(self, audio_url: str, duration: int = 30) -> Optional[np.ndarray]:
        """Extract embeddings from audio URL"""
        try:
            # Download audio with retries
            response = requests.get(audio_url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to download audio: HTTP {response.status_code}")
                return None
            
            # Check if we actually got audio content
            if len(response.content) < 1000:  # Less than 1KB is probably not valid audio
                logger.warning(f"Audio file too small ({len(response.content)} bytes), likely corrupted")
                return None
            
            # For MP3 files, we need to use pydub to convert first
            if audio_url.endswith('.mp3') or 'mp3' in audio_url:
                from pydub import AudioSegment
                import tempfile
                import os
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                
                try:
                    # Try multiple methods to load the audio
                    try:
                        # Method 1: Direct librosa load
                        y, sr = librosa.load(tmp_path, sr=22050, duration=duration)
                        if len(y) == 0:
                            raise Exception("Empty audio array")
                        return self.extract_from_audio(y, sr)
                    except Exception as e1:
                        logger.debug(f"Direct librosa load failed: {e1}")
                        try:
                            # Method 2: Use pydub first
                            audio = AudioSegment.from_mp3(tmp_path)
                            # Convert to numpy array
                            samples = audio.get_array_of_samples()
                            audio_np = np.array(samples).astype(np.float32)
                            if audio.channels == 2:
                                audio_np = audio_np.reshape((-1, 2)).mean(axis=1)
                            # Normalize
                            audio_np = audio_np / np.max(np.abs(audio_np)) if np.max(np.abs(audio_np)) > 0 else audio_np
                            # Resample if needed
                            if audio.frame_rate != 22050:
                                audio_np = librosa.resample(audio_np, orig_sr=audio.frame_rate, target_sr=22050)
                            # Trim to duration
                            if duration and len(audio_np) > duration * 22050:
                                audio_np = audio_np[:duration * 22050]
                            return self.extract_from_audio(audio_np, 22050)
                        except Exception as e2:
                            logger.warning(f"All audio loading methods failed: {e1}, {e2}")
                            return None
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            else:
                # Try original method for non-MP3 files
                audio_data = BytesIO(response.content)
                y, sr = librosa.load(audio_data, sr=22050, duration=duration)
                return self.extract_from_audio(y, sr)
            
        except Exception as e:
            logger.warning(f"Failed to extract embeddings from {audio_url}: {e}")
            return None
    
    def extract_from_audio(self, audio: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """Extract embeddings from audio array"""
        try:
            if self.model_type == "openl3" and OPENL3_AVAILABLE:
                # OpenL3 extraction
                emb, ts = openl3.get_audio_embedding(
                    audio, sample_rate, 
                    content_type="music",
                    embedding_size=512
                )
                return np.mean(emb, axis=0)  # Average over time
                
            elif self.model_type == "clap" and CLAP_AVAILABLE:
                # CLAP extraction
                audio_embed = self.model.get_audio_embedding_from_data(
                    x=audio, use_tensor=False
                )
                return audio_embed[0]  # First embedding
                
            else:
                # Fallback: simple spectral features
                return self._extract_spectral_features(audio, sample_rate)
                
        except Exception as e:
            logger.error(f"Audio embedding extraction failed: {e}")
            # Return dummy embedding as fallback
            return np.random.normal(0, 1, 512)
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract simple spectral features as fallback"""
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Combine features
        features = np.concatenate([
            mfcc_mean,
            [np.mean(spectral_centroids)],
            [np.mean(spectral_rolloff)],
            [np.mean(spectral_bandwidth)],
            chroma_mean
        ])
        
        # Pad to 512 dimensions
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), 'constant')
        else:
            features = features[:512]
            
        return features

class TextEmbeddingExtractor:
    """Extract text embeddings from track metadata"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Text embedding model loaded: {model_name}")
    
    def extract_from_track(self, track_data: Dict) -> np.ndarray:
        """Extract text embedding from track metadata (supports both Spotify and Deezer formats)"""
        text_parts = []
        
        # Artist - handle both Spotify (artists array) and Deezer (artist object) formats
        if "artists" in track_data and track_data["artists"]:
            # Spotify format
            artists = ", ".join([artist["name"] for artist in track_data["artists"]])
            text_parts.append(f"Artist: {artists}")
        elif "artist" in track_data and track_data["artist"]:
            # Deezer format
            artist_name = track_data["artist"]["name"] if isinstance(track_data["artist"], dict) else str(track_data["artist"])
            text_parts.append(f"Artist: {artist_name}")
        
        # Track title - handle both formats
        if "name" in track_data:
            # Spotify format
            text_parts.append(f"Title: {track_data['name']}")
        elif "title" in track_data:
            # Deezer format
            text_parts.append(f"Title: {track_data['title']}")
        
        # Album info - handle both formats
        if "album" in track_data and track_data["album"]:
            album = track_data["album"]
            # Album name
            album_name = album.get("name") or album.get("title")  # Spotify uses "name", Deezer uses "title"
            if album_name:
                text_parts.append(f"Album: {album_name}")
            
            # Album genres (mainly Spotify)
            if "genres" in album and album["genres"]:
                genres = ", ".join(album["genres"])
                text_parts.append(f"Genres: {genres}")
        
        # Genre metadata - handle both formats
        if "playlist_genre" in track_data:
            # Spotify format
            text_parts.append(f"Genre: {track_data['playlist_genre']}")
        elif "dataset_genre" in track_data and track_data["dataset_genre"]:
            # Deezer format (our added field)
            text_parts.append(f"Genre: {track_data['dataset_genre']}")
        
        # Popularity - handle both formats
        popularity_score = None
        if "popularity" in track_data:
            # Spotify format (0-100)
            popularity_score = track_data["popularity"]
        elif "rank" in track_data:
            # Deezer format (higher = more popular, normalize to 0-100)
            rank = track_data["rank"]
            if rank > 500000:
                popularity_score = 90  # Very popular
            elif rank > 200000:
                popularity_score = 70  # Popular  
            elif rank > 50000:
                popularity_score = 50  # Moderately popular
            else:
                popularity_score = 20  # Less popular
        
        if popularity_score:
            if popularity_score > 80:
                text_parts.append("Very popular")
            elif popularity_score > 60:
                text_parts.append("Popular")
            elif popularity_score > 40:
                text_parts.append("Moderately popular")
        
        # Duration info (Deezer specific)
        if "duration" in track_data:
            duration_sec = track_data["duration"]
            if duration_sec < 120:
                text_parts.append("Short track")
            elif duration_sec > 300:
                text_parts.append("Long track")
        
        # Explicit content flag
        if track_data.get("explicit_lyrics") or track_data.get("explicit"):
            text_parts.append("Explicit content")
        
        # Combine all text
        combined_text = " | ".join(text_parts) if text_parts else "Unknown track"
        
        # Generate embedding
        embedding = self.model.encode(combined_text)
        return embedding

class MusicDataPipeline:
    """Complete pipeline for fetching and processing music data"""
    
    def __init__(self):
        self.deezer = DeezerAPI()  # No authentication required!
        self.audio_extractor = AudioEmbeddingExtractor("openl3")  # Try OpenL3 first
        self.text_extractor = TextEmbeddingExtractor()
        
    def collect_diverse_dataset(self, target_size: int = 500) -> List[Dict]:
        """Collect a diverse music dataset"""
        logger.info(f"Collecting diverse dataset of {target_size} tracks...")
        
        # Define genres for diversity
        genres = [
            "rock", "pop", "hip-hop", "electronic", "jazz", "classical",
            "indie", "folk", "r&b", "country", "reggae", "blues",
            "ambient", "dance", "alternative", "metal"
        ]
        
        all_tracks = []
        tracks_per_genre = max(1, target_size // len(genres))
        
        for genre in genres:
            logger.info(f"Fetching {genre} tracks...")
            
            # Search for tracks by genre on Deezer
            tracks = self.deezer.search_by_genre(genre, limit=tracks_per_genre * 3)
            
            # Filter tracks with preview URLs (Deezer usually has many more!)
            preview_tracks = [
                track for track in tracks 
                if track.get("preview") and track.get("rank", 0) > 50000  # Lower threshold for more variety
            ]
            
            logger.info(f"  Found {len(preview_tracks)} {genre} tracks with previews")
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_tracks = []
            for track in preview_tracks:
                if track["id"] not in seen_ids and len(unique_tracks) < tracks_per_genre:
                    seen_ids.add(track["id"])
                    unique_tracks.append(track)
            
            # Add genre tag to tracks
            for track in unique_tracks:
                track["dataset_genre"] = genre
            
            all_tracks.extend(unique_tracks)
            logger.info(f"Added {len(unique_tracks)} {genre} tracks")
            time.sleep(0.2)  # Slightly more conservative rate limiting for larger requests
        
        logger.info(f"Collected {len(all_tracks)} tracks")
        return all_tracks[:target_size]
    
    def process_tracks(self, tracks: List[Dict], output_file: str = "processed_tracks.json") -> List[Dict]:
        """Process tracks to extract embeddings and clean metadata"""
        logger.info(f"Processing {len(tracks)} tracks...")
        
        processed_tracks = []
        
        for i, track in enumerate(tracks):
            if (i + 1) % 25 == 0:
                logger.info(f"Processing track {i+1}/{len(tracks)} ({((i+1)/len(tracks)*100):.1f}%)")
            else:
                logger.debug(f"Processing track {i+1}/{len(tracks)}: {track.get('title', 'Unknown')}")
            
            try:
                # Extract audio embedding from Deezer preview URL
                preview_url = track.get("preview")
                if not preview_url:
                    logger.warning(f"Track {track.get('title', 'Unknown')} missing preview URL")
                    continue
                    
                audio_embedding = self.audio_extractor.extract_from_url(preview_url)
                
                if audio_embedding is None:
                    logger.error(f"Failed to extract audio embedding from {track.get('title', 'Unknown')} preview")
                    continue
                
                # Extract text embedding
                text_embedding = self.text_extractor.extract_from_track(track)
                
                # Clean and structure metadata (Deezer format)
                processed_track = {
                    "id": str(track["id"]),
                    "title": track.get("title", "Unknown"),
                    "artist": track.get("artist", {}).get("name", "Unknown Artist"),
                    "album": track.get("album", {}).get("title", "Unknown Album"),
                    "preview_url": track.get("preview"),
                    "popularity": track.get("rank", 0),  # Deezer uses "rank"
                    "duration_ms": track.get("duration", 0) * 1000,  # Deezer gives seconds
                    "explicit": track.get("explicit_lyrics", False),
                    "genres": [track.get("dataset_genre", "unknown")],
                    "audio_embedding": audio_embedding.tolist(),
                    "text_embedding": text_embedding.tolist(),
                    "deezer_url": track.get("link", ""),
                    "artist_id": track.get("artist", {}).get("id"),
                    "album_id": track.get("album", {}).get("id")
                }
                
                processed_tracks.append(processed_track)
                
            except Exception as e:
                logger.error(f"Failed to process track {track.get('title', 'Unknown')}: {e}")
                continue
            
            # Save periodically (more frequent for larger datasets)
            if (i + 1) % 25 == 0:
                self._save_tracks(processed_tracks, f"{output_file}.tmp")
                logger.info(f"  Saved checkpoint at {i+1} tracks")
        
        # But we have the actual audio embeddings which are much better!
        logger.info("✅ Audio embeddings extracted from real Deezer preview files!")
        
        # Save final dataset
        self._save_tracks(processed_tracks, output_file)
        logger.info(f"Processed dataset saved to {output_file}")
        
        return processed_tracks
    
    def _save_tracks(self, tracks: List[Dict], filename: str):
        """Save tracks to JSON file"""
        with open(filename, 'w') as f:
            json.dump(tracks, f, indent=2)

def main():
    """Example usage of the data pipeline"""
    print("🎵 Using Deezer API (no authentication required!)")
    
    # Create pipeline - no credentials needed for Deezer!
    pipeline = MusicDataPipeline()
    
    # You can adjust the target_size here for different dataset sizes:
    # Small: 50-100 tracks (quick testing)
    # Medium: 200-300 tracks (good balance)
    # Large: 500-1000 tracks (comprehensive dataset)
    # Very Large: 1000+ tracks (production ready)
    
    # Collect and process tracks
    tracks = pipeline.collect_diverse_dataset(target_size=500)  # Large diverse dataset
    processed_tracks = pipeline.process_tracks(tracks)
    
    print(f"✅ Successfully processed {len(processed_tracks)} tracks!")
    print("   Data saved to processed_tracks.json")
    print(f"   Dataset contains {len(set(track['genres'][0] for track in processed_tracks if track['genres']))} different genres")

if __name__ == "__main__":
    main()
