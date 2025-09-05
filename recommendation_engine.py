import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self, sentence_model, genre_centroids: Dict, tracks_db: Dict):
        self.sentence_model = sentence_model
        self.genre_centroids = genre_centroids
        self.tracks_db = tracks_db
        
        # Initialize data loader for genre information
        from models_real import MusicDataLoader
        self.data_loader = MusicDataLoader()
    
    def parse_user_intent(self, description: str) -> Tuple[Dict[str, float], np.ndarray]:
        """Parse user description using LLM-based genre confidence scoring"""
        
        # Get available genres from our centroids
        available_genres = list(self.genre_centroids.keys())
        
        # Create genre confidence prompt for LLM with dynamic context
        prompt = self._build_genre_prompt(description, available_genres)
        
        # Use LLM for genre confidence scoring
        genre_scores = self._compute_llm_genre_scores(prompt, available_genres)
        
        # Create text embedding for the user description
        text_embedding = self.sentence_model.encode([description])[0]
        
        logger.info(f"LLM-based genre confidence scores for '{description}': {genre_scores}")
        return genre_scores, text_embedding
    
    def _build_genre_prompt(self, description: str, available_genres: List[str]) -> str:
        """Build dynamic genre prompt using information from models_real.py"""
        
        # Group genres by energy level
        calm_genres = []
        moderate_genres = []
        energetic_genres = []
        
        for genre in available_genres:
            energy = self.data_loader._get_genre_energy_level(genre)
            if energy == "calm":
                calm_genres.append(genre)
            elif energy == "moderate":
                moderate_genres.append(genre)
            elif energy == "energetic":
                energetic_genres.append(genre)
        
        # Build genre descriptions
        def format_genre_list(genres):
            return "\n".join([
                f"- {genre}: {self.data_loader._get_genre_description(genre)} (Activities: {', '.join(self.data_loader._get_genre_activities(genre)[:3])})"
                for genre in genres
            ])
        
        prompt = f"""
Given the user's music request: "{description}"

Please analyze this request and provide confidence scores (0.0 to 1.0) for how well each music genre would satisfy the user's needs.

Here are the available genres with their characteristics:
"""
        
        if calm_genres:
            prompt += f"""
**CALM/RELAXING GENRES:**
{format_genre_list(calm_genres)}
"""
        
        if moderate_genres:
            prompt += f"""
**MODERATE ENERGY GENRES:**
{format_genre_list(moderate_genres)}
"""
        
        if energetic_genres:
            prompt += f"""
**ENERGETIC/UPBEAT GENRES:**
{format_genre_list(energetic_genres)}
"""
        
        # Add activity-based guidance
        activity_mapping = {
            "study/focus": [g for g in available_genres if "study" in self.data_loader._get_genre_activities(g) or "focus" in self.data_loader._get_genre_activities(g)],
            "workout/energy": [g for g in available_genres if "workout" in self.data_loader._get_genre_activities(g) or "energy" in self.data_loader._get_genre_activities(g)],
            "party/social": [g for g in available_genres if "parties" in self.data_loader._get_genre_activities(g) or "dancing" in self.data_loader._get_genre_activities(g)],
            "relaxation": [g for g in available_genres if "relaxation" in self.data_loader._get_genre_activities(g)],
            "emotional": [g for g in available_genres if "emotional" in " ".join(self.data_loader._get_genre_activities(g))]
        }
        
        if any(activity_mapping.values()):
            prompt += "\n**ACTIVITY MATCHING:**\n"
            for activity, genres in activity_mapping.items():
                if genres:
                    prompt += f"- {activity.title()}: {', '.join(genres)} (recommended scores: 0.7-0.9)\n"
        
        prompt += f"""
Analyze the user's request for mood, energy level, and context. Respond with ONLY a JSON object:

{{"genre_name": confidence_score}}

User request: "{description}"
"""
        
        return prompt
    
    def _compute_llm_genre_scores(self, prompt: str, available_genres: List[str]) -> Dict[str, float]:
        """Compute genre confidence scores using LLM API (Mistral/OpenAI/Anthropic)"""
        import json
        
        # Try Mistral first (priority)
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            try:
                from mistralai import Mistral
                
                client = Mistral(api_key=mistral_key)
                
                response = client.chat.complete(
                    model="mistral-small",  # or "mistral-medium", "mistral-large"
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                
                response_text = response.choices[0].message.content.strip()
                genre_scores = self._parse_llm_response(response_text, "Mistral")
                
                # Validate and normalize scores
                validated_scores = self._validate_genre_scores(genre_scores, available_genres, "Mistral")
                
                logger.info("✅ Using Mistral AI for genre scoring")
                return validated_scores
                
            except Exception as e:
                logger.warning(f"Mistral AI API failed: {e}")
        
        # Try OpenAI second
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                import openai
                openai.api_key = openai_key
                
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                
                response_text = response.choices[0].message.content.strip()
                genre_scores = self._parse_llm_response(response_text, "OpenAI")
                
                # Validate and normalize scores
                validated_scores = self._validate_genre_scores(genre_scores, available_genres, "OpenAI")
                
                logger.info("✅ Using OpenAI GPT for genre scoring")
                return validated_scores
                
            except Exception as e:
                logger.warning(f"OpenAI API failed: {e}")
        
        # Try Anthropic Claude third
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") 
        if anthropic_key:
            try:
                import anthropic
                
                client = anthropic.Anthropic(api_key=anthropic_key)
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response_text = message.content[0].text.strip()
                genre_scores = self._parse_llm_response(response_text, "Anthropic")
                
                # Validate and normalize scores
                validated_scores = self._validate_genre_scores(genre_scores, available_genres, "Anthropic")
                
                logger.info("✅ Using Anthropic Claude for genre scoring")
                return validated_scores
                
            except Exception as e:
                logger.warning(f"Anthropic API failed: {e}")
        
        # If no LLM APIs available, raise clear error
        raise Exception("No LLM API keys found! Please set one of: MISTRAL_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
    
    def _parse_llm_response(self, response_text: str, provider: str) -> Dict[str, float]:
        """Robustly parse LLM response to extract JSON, handling common formatting issues"""
        import json
        import re
        
        logger.debug(f"{provider} raw response: {response_text[:200]}...")
        
        try:
            # First, try direct JSON parsing
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.warning(f"{provider} direct JSON parsing failed: {e}")
            
            # Try to extract JSON from the response using regex
            # Look for content between curly braces
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response_text)
            
            for match in json_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # Try to find and clean JSON-like content
            # Remove markdown code blocks
            cleaned = re.sub(r'```(?:json|JSON)?\s*', '', response_text)
            cleaned = re.sub(r'```\s*', '', cleaned)
            
            # Try to extract JSON from code blocks first
            code_block_pattern = r'```(?:json|JSON)?\s*(\{[^`]+\})\s*```'
            code_match = re.search(code_block_pattern, response_text, re.DOTALL)
            if code_match:
                try:
                    return json.loads(code_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Remove extra text before and after JSON
            lines = cleaned.split('\n')
            json_lines = []
            in_json = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('{'):
                    in_json = True
                    json_lines.append(line)
                elif in_json:
                    json_lines.append(line)
                    if line.endswith('}'):
                        break
            
            if json_lines:
                try:
                    json_text = '\n'.join(json_lines)
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass
            
            # Final fallback: try to parse line by line for key-value pairs
            logger.warning(f"{provider} JSON extraction failed, attempting manual parsing")
            
            # Look for patterns like "genre": 0.8, 'genre': 0.8, or genre: 0.8
            genre_patterns = [
                r'["\']([^"\']+)["\']:\s*([0-9.]+)',  # "genre": 0.8
                r'([a-zA-Z-]+):\s*([0-9.]+)',         # genre: 0.8
                r'([a-zA-Z-]+)\s*=\s*([0-9.]+)',     # genre = 0.8
            ]
            
            matches = []
            for pattern in genre_patterns:
                pattern_matches = re.findall(pattern, response_text)
                if pattern_matches:
                    matches.extend(pattern_matches)
                    break  # Use first successful pattern
            
            if matches:
                result = {}
                for genre, score in matches:
                    try:
                        result[genre] = float(score)
                    except ValueError:
                        continue
                        
                if result:
                    logger.info(f"{provider} manual parsing successful, found {len(result)} genres")
                    return result
            
            # If all else fails, log the response and raise
            logger.error(f"{provider} failed to parse response: {response_text}")
            raise Exception(f"{provider} returned unparseable response. Raw content: {response_text[:100]}...")
    
    def _validate_genre_scores(self, genre_scores: Dict[str, float], available_genres: List[str], provider: str) -> Dict[str, float]:
        """Validate and normalize genre scores, with basic sanity checks"""
        
        # Basic validation and normalization
        validated_scores = {}
        for genre in available_genres:
            score = genre_scores.get(genre, 0.01)
            validated_scores[genre] = max(0.0, min(1.0, float(score)))
        
        # Normalize to sum to 1.0
        total = sum(validated_scores.values())
        if total > 0:
            validated_scores = {g: s/total for g, s in validated_scores.items()}
        
        # Log the top genres for debugging
        top_genres = sorted(validated_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"{provider} top genres: {[(g, f'{s:.3f}') for g, s in top_genres]}")
        
        return validated_scores
            
    def compute_genre_weighted_audio_similarity(self, genre_scores: Dict[str, float], track_embeddings: np.ndarray) -> np.ndarray:
        """Compute weighted audio similarity based on genre confidence scores"""
        total_similarities = np.zeros(len(track_embeddings))
        
        for genre, confidence in genre_scores.items():
            if genre in self.genre_centroids and confidence > 0:
                genre_centroid = np.array(self.genre_centroids[genre]["audio_centroid"]).reshape(1, -1)
                similarities = cosine_similarity(genre_centroid, track_embeddings)[0]
                total_similarities += confidence * similarities
        
        return total_similarities
    
    def compute_text_similarity(self, user_embedding: np.ndarray, track_embeddings: np.ndarray) -> np.ndarray:
        """Compute text similarity between user description and tracks"""
        user_embedding = user_embedding.reshape(1, -1)
        similarities = cosine_similarity(user_embedding, track_embeddings)[0]
        return similarities
    
    def fusion_reranking(self, audio_scores: np.ndarray, text_scores: np.ndarray, 
                        alpha: float = 0.8) -> np.ndarray:
        """Combine audio and text similarity scores using weighted fusion"""
        # Normalize scores to [0, 1] range
        audio_scores_norm = (audio_scores - audio_scores.min()) / (audio_scores.max() - audio_scores.min() + 1e-8)
        text_scores_norm = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min() + 1e-8)
        
        # Weighted combination (alpha for audio, 1-alpha for text)
        fused_scores = alpha * audio_scores_norm + (1 - alpha) * text_scores_norm
        return fused_scores
    
    def generate_explanation(self, track: Dict, genre_scores: Dict[str, float], similarity_score: float) -> str:
        """Generate explanation for why a track was recommended"""
        track_genre = track["genres"][0] if track["genres"] else "unknown"
        
        # Find the top genre from user's intent that matches this track
        matching_genres = [genre for genre in genre_scores.keys() if genre == track_genre]
        top_genre = matching_genres[0] if matching_genres else max(genre_scores.keys(), key=genre_scores.get)
        confidence = genre_scores.get(top_genre, 0)
        
        explanations = [
            f"Perfect {track_genre} match with {confidence:.1%} confidence based on your request.",
            f"This {track_genre} track aligns well with your preferences for {top_genre} music.",
            f"Strong {track_genre} recommendation - detected {confidence:.1%} genre match.",
            f"Excellent {track_genre} choice based on your musical taste preferences."
        ]
        
        # Choose explanation based on similarity score
        explanation_idx = min(int(similarity_score * len(explanations)), len(explanations) - 1)
        return explanations[explanation_idx]
    
    def recommend_tracks(self, description: str, max_tracks: int = 10) -> Dict:
        """Main recommendation function using genre-based recommendations"""
        try:
            # Parse user intent to get genre confidence scores
            genre_scores, user_text_embedding = self.parse_user_intent(description)
            
            # Prepare track data
            track_ids = list(self.tracks_db.keys())
            track_audio_embeddings = np.array([
                self.tracks_db[track_id]["audio_embedding"] for track_id in track_ids
            ])
            track_text_embeddings = np.array([
                self.tracks_db[track_id]["text_embedding"] for track_id in track_ids
            ])
            
            # Compute similarities using genre-weighted approach
            audio_similarities = self.compute_genre_weighted_audio_similarity(genre_scores, track_audio_embeddings)
            text_similarities = self.compute_text_similarity(user_text_embedding, track_text_embeddings)
            
            # Fusion reranking
            fused_scores = self.fusion_reranking(audio_similarities, text_similarities)
            
            # Get top tracks
            top_indices = np.argsort(fused_scores)[::-1][:max_tracks]
            
            # Prepare recommendations
            recommendations = []
            for idx in top_indices:
                track_id = track_ids[idx]
                track = self.tracks_db[track_id]
                similarity_score = float(fused_scores[idx])
                explanation = self.generate_explanation(track, genre_scores, similarity_score)
                
                recommendations.append({
                    "track": track,
                    "similarity_score": similarity_score,
                    "explanation": explanation
                })
            
            # Get primary genre for response
            top_genre = max(genre_scores.keys(), key=genre_scores.get) if genre_scores else "mixed"
            
            return {
                "tracks": recommendations,
                "genre_detected": top_genre,
                "genre_scores": genre_scores,
                "total_tracks": len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error in recommendation engine: {str(e)}")
            raise e
