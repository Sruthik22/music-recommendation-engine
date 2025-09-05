import requests
import json

# Test client for the music recommendation API
API_BASE_URL = "http://localhost:8000"

def test_api():
    """Test the music recommendation API"""
    
    print("🎵 Testing Music Recommendation Engine API\n")
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test available moods
    print("2. Getting available moods...")
    try:
        response = requests.get(f"{API_BASE_URL}/moods")
        print(f"   Status: {response.status_code}")
        print(f"   Available moods: {response.json()['moods']}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test sample tracks
    print("3. Getting sample tracks...")
    try:
        response = requests.get(f"{API_BASE_URL}/tracks/sample")
        print(f"   Status: {response.status_code}")
        sample_tracks = response.json()['sample_tracks']
        print(f"   Sample tracks:")
        for track in sample_tracks:
            print(f"     - {track['title']} by {track['artist']} (Tags: {', '.join(track['tags'])})")
        print()
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test recommendations with different descriptions
    test_descriptions = [
        "I want something chill and relaxing for studying",
        "Give me energetic workout music",
        "I'm feeling sad and need emotional songs",
        "Party music that makes me feel euphoric",
        "Background ambient music for meditation"
    ]
    
    print("4. Testing recommendations...")
    for i, description in enumerate(test_descriptions, 1):
        print(f"   Test {i}: '{description}'")
        try:
            payload = {
                "description": description,
                "max_tracks": 5
            }
            response = requests.post(f"{API_BASE_URL}/recommend", json=payload)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   Detected mood: {result['mood_detected']}")
                print(f"   Recommended tracks:")
                for track_rec in result['tracks']:
                    track = track_rec['track']
                    score = track_rec['similarity_score']
                    explanation = track_rec['explanation']
                    print(f"     - {track['title']} by {track['artist']} (Score: {score:.3f})")
                    print(f"       {explanation}")
            else:
                print(f"   Error: {response.text}")
            print()
        except Exception as e:
            print(f"   Error: {e}\n")

if __name__ == "__main__":
    test_api()
