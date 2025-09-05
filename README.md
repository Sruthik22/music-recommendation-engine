# Music Recommendation Engine

A Python FastAPI application that provides personalized music recommendations based on user descriptions using real audio embeddings, LLM-powered genre confidence scoring, and fusion reranking.

## Features

-   **Real Audio Data**: Uses actual music tracks from Deezer API with 30-second previews
-   **Advanced Audio Embeddings**: Extracts features using OpenL3, CLAP, or spectral analysis
-   **LLM-Powered Genre Detection**: Uses Mistral AI, OpenAI, or Anthropic for intelligent genre confidence scoring
-   **Dual Embedding System**: Combines audio and text embeddings for comprehensive matching
-   **Fusion Reranking**: Sophisticated scoring system that weights audio and text similarity
-   **Diverse Music Database**: 500+ tracks across 16 different genres with real metadata
-   **Explanatory Responses**: Provides clear explanations for why each track was recommended
-   **RESTful API**: Clean FastAPI endpoints with automatic documentation

## Quick Start

### Prerequisites

-   **Python 3.9+**
-   **FFmpeg** (for audio processing)
-   **LLM API Key** (Mistral AI, OpenAI, or Anthropic)

### Installation

#### Option 1: Using uv (Recommended - Fast & Modern)

1. **Install uv if you haven't already:**

    ```bash
    # On macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

    # Or with pip
    pip install uv
    ```

2. **Clone and navigate to the project directory:**

    ```bash
    cd music-recommendation-engine
    ```

3. **Install dependencies and create virtual environment:**

    ```bash
    uv sync
    ```

4. **Set up environment variables:**
    ```bash
    cp env.template .env
    # Edit .env and add your LLM API key
    ```

#### Option 2: Using pip (Traditional)

1. **Clone and navigate to the project directory:**

    ```bash
    cd music-recommendation-engine
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -e .
    ```

4. **Set up environment variables:**
    ```bash
    cp env.template .env
    # Edit .env and add your LLM API key
    ```

### Initial Setup (First Time Only)

1. **Collect music data:**

    ```bash
    # With uv
    uv run python data_pipeline.py

    # With pip/venv
    python data_pipeline.py
    ```

    This will collect 500 diverse music tracks with real audio embeddings from Deezer (takes ~10-15 minutes).

### Running the Application

1. **Start the FastAPI server:**

    **With uv:**

    ```bash
    uv run python main.py
    ```

    or use the dedicated run script:

    ```bash
    uv run python run.py
    ```

    **With pip/venv:**

    ```bash
    python main.py
    ```

    or

    ```bash
    python run.py
    ```

2. **Access the API:**
    - API Base URL: `http://localhost:8000`
    - Interactive Documentation: `http://localhost:8000/docs`
    - Alternative Documentation: `http://localhost:8000/redoc`

### Testing the API

Run the test client to see the recommendation engine in action:

**With uv:**

```bash
uv run python test_client.py
```

**With pip/venv:**

```bash
python test_client.py
```

## API Endpoints

### Core Endpoints

-   **`GET /`** - Root endpoint with API information
-   **`GET /health`** - Health check
-   **`GET /genres`** - List available music genres
-   **`POST /recommend`** - Generate playlist recommendations
-   **`GET /tracks/sample`** - Get sample tracks from the database
-   **`GET /stats`** - Get database statistics

### Recommendation Request

```json
{
	"description": "I want something chill and relaxing for studying",
	"max_tracks": 10
}
```

### Recommendation Response

```json
{
	"tracks": [
		{
			"track": {
				"id": "123456789",
				"title": "Ocean Waves",
				"artist": "Calm Collective",
				"album": "Peaceful Moments",
				"genres": ["ambient"],
				"preview_url": "https://cdn-preview-1.deezer.com/stream/...",
				"duration_ms": 180000,
				"popularity": 75,
				"explicit": false
			},
			"audio_similarity": 0.89,
			"text_similarity": 0.82,
			"combined_score": 0.86,
			"explanation": "Perfect ambient track with soothing qualities, ideal for studying"
		}
	],
	"query_analysis": {
		"genre_confidence": {
			"ambient": 0.8,
			"electronic": 0.6,
			"classical": 0.4
		},
		"detected_intent": "relaxing background music for concentration"
	},
	"total_tracks": 10
}
```

## Architecture

### Components

1. **Main Application (`main.py`)**

    - FastAPI app setup and route definitions
    - Middleware configuration
    - Real-time recommendation serving

2. **Data Pipeline (`data_pipeline.py`)**

    - Deezer API integration for real music data
    - Audio embedding extraction (OpenL3/CLAP/spectral)
    - Text embedding generation
    - Dataset collection and processing

3. **Data Models (`models_real.py`)**

    - Real track database with audio/text embeddings
    - Genre centroids from actual music data
    - Database loading and management

4. **Recommendation Engine (`recommendation_engine.py`)**

    - LLM-powered genre confidence scoring
    - Advanced similarity computation
    - Fusion reranking algorithm
    - Intelligent explanation generation

5. **Demo & Testing (`demo.py`, `test_client.py`)**
    - Interactive demonstration scripts
    - API testing and validation

### Algorithm Flow

1. **LLM Genre Analysis**: Send user description to LLM for genre confidence scoring
2. **Audio Similarity**: Compare genre-weighted centroids with track audio embeddings
3. **Text Similarity**: Compare user description with track text embeddings using sentence transformers
4. **Fusion Reranking**: Intelligently combine audio and text scores with dynamic weighting
5. **Explanation Generation**: Create contextual explanations for each recommendation
6. **Result Ranking**: Return top-scored tracks with detailed metadata

## Available Genres

The system includes 16 diverse music genres with real tracks and metadata:

-   **Rock**: Classic and modern rock tracks
-   **Pop**: Contemporary pop music
-   **Hip-hop**: Rap and hip-hop tracks
-   **Electronic**: Electronic dance music and synthwave
-   **Jazz**: Traditional and contemporary jazz
-   **Classical**: Orchestral and chamber music
-   **Indie**: Independent and alternative music
-   **Folk**: Acoustic and traditional folk music
-   **R&B**: Rhythm and blues, soul music
-   **Country**: Country and folk-country music
-   **Reggae**: Reggae and Caribbean music
-   **Blues**: Traditional and electric blues
-   **Ambient**: Atmospheric and ambient soundscapes
-   **Dance**: Dance floor and club music
-   **Alternative**: Alternative rock and indie
-   **Metal**: Heavy metal and hard rock

## Example Usage

### Curl Examples

```bash
# Get available genres
curl -X GET "http://localhost:8000/genres"

# Get database statistics
curl -X GET "http://localhost:8000/stats"

# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"description": "upbeat workout music", "max_tracks": 5}'

# Get sample tracks
curl -X GET "http://localhost:8000/tracks/sample?limit=10"

# Health check
curl -X GET "http://localhost:8000/health"
```

### Python Client Example

```python
import requests

# Get recommendations
response = requests.post("http://localhost:8000/recommend", json={
    "description": "chill study music with ambient vibes",
    "max_tracks": 8
})

recommendations = response.json()
print(f"Genre confidence: {recommendations['query_analysis']['genre_confidence']}")

for track_rec in recommendations['tracks']:
    track = track_rec['track']
    print(f"- {track['title']} by {track['artist']}")
    print(f"  Score: {track_rec['combined_score']:.2f}")
    print(f"  {track_rec['explanation']}")
    if track['preview_url']:
        print(f"  Preview: {track['preview_url']}")
```

## Extending the System

### Adding More Music Data

1. **Increase Dataset Size**: Modify `target_size` in `data_pipeline.py` for more tracks:

    ```python
    tracks = pipeline.collect_diverse_dataset(target_size=1000)  # 1000 tracks
    ```

2. **Add New Genres**: Extend the genres list in `data_pipeline.py`:

    ```python
    genres = ["rock", "pop", "your_new_genre", ...]
    ```

3. **Custom Audio Models**: Replace OpenL3 with other models in `AudioEmbeddingExtractor`

### Customizing LLM Integration

1. **Switch LLM Provider**: Update API keys in `.env` file
2. **Modify Genre Confidence**: Adjust scoring logic in `recommendation_engine.py`
3. **Add New LLM Providers**: Extend the LLM client handling

### Performance Optimization

1. **Database Integration**: Replace JSON files with PostgreSQL/MongoDB
2. **Vector Search**: Use Pinecone, Weaviate, or Faiss for large-scale similarity search
3. **Caching**: Add Redis for API response caching
4. **Async Processing**: Implement async audio processing for larger datasets

## Dependencies

### Core Framework

-   **FastAPI**: Web framework for building APIs
-   **uvicorn**: ASGI server for FastAPI
-   **pydantic**: Data validation and serialization

### Machine Learning

-   **sentence-transformers**: Text embedding generation
-   **scikit-learn**: Cosine similarity computation
-   **numpy**: Numerical operations
-   **pandas**: Data manipulation

### Audio Processing

-   **librosa**: Audio analysis and feature extraction
-   **openl3**: Pre-trained audio embeddings (optional)
-   **pydub**: Audio file format conversion
-   **ffmpeg-python**: Audio preprocessing

### LLM Integration

-   **mistralai**: Mistral AI API client
-   **openai**: OpenAI API client
-   **anthropic**: Anthropic API client

### Data Collection

-   **requests**: HTTP client for Deezer API
-   **python-dotenv**: Environment variable management

## Performance Notes

-   **Audio Embeddings**: 512-dimensional vectors from OpenL3/CLAP/spectral features
-   **Text Embeddings**: 384-dimensional vectors from sentence-transformers
-   **Memory Usage**: ~50MB for 500 tracks with full embeddings
-   **Response Time**: <200ms for recommendations on 500-track database
-   **Scalability**: For >5k tracks, consider vector databases (Pinecone, Weaviate, Faiss)
-   **LLM Calls**: Cached for 1 hour to minimize API costs

## Project Structure

```
music-recommendation-engine/
├── main.py                 # FastAPI application
├── data_pipeline.py        # Data collection from Deezer API
├── models_real.py          # Data models and loading
├── recommendation_engine.py # Core recommendation logic
├── demo.py                 # Interactive demo script
├── test_client.py          # API testing client
├── run.py                  # Startup script
├── env.template            # Environment variables template
├── pyproject.toml          # Dependencies and project config
├── uv.lock                 # Dependency lock file
├── processed_tracks.json   # Music database (generated)
├── dataset_summary.json    # Dataset statistics (generated)
└── README.md               # This file
```

## License

MIT License - Feel free to use and modify for your projects!

## 🧠 LLM Integration Setup

The music recommendation engine uses LLMs for intelligent genre confidence scoring.

### Supported LLM Providers (in order of priority):

1. **Mistral AI** (Primary) - Most cost-effective and fast
2. **OpenAI** (Fallback) - GPT-3.5-turbo
3. **Anthropic** (Fallback) - Claude-3-haiku

### Quick Setup:

1. Copy the environment template:

    ```bash
    cp env.template .env
    ```

2. Add your API key to `.env`:

    ```bash
    # For Mistral (recommended)
    MISTRAL_API_KEY=your_actual_mistral_key

    # OR for OpenAI
    OPENAI_API_KEY=your_actual_openai_key

    # OR for Anthropic
    ANTHROPIC_API_KEY=your_actual_anthropic_key
    ```

3. Test the system:
    ```bash
    uv run python demo.py
    ```

### Getting API Keys:

-   **Mistral**: https://console.mistral.ai/
-   **OpenAI**: https://platform.openai.com/api-keys
-   **Anthropic**: https://console.anthropic.com/

### How It Works:

The LLM analyzes user requests like "I want heavy metal for working out" and returns confidence scores:

```json
{ "metal": 0.8, "rock": 0.6, "electronic": 0.3, "pop": 0.1 }
```

These scores weight the audio similarity calculations for more accurate recommendations.
