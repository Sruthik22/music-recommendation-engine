#!/usr/bin/env python3
"""
Startup script for the Music Recommendation Engine
"""

import uvicorn
import sys
import os
import subprocess

def check_uv_available():
    """Check if uv is available in the system"""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    """Main function to start the FastAPI application"""
    
    print("🎵 Starting Music Recommendation Engine...")
    print("=" * 50)
    
    # Check if we're running with uv
    is_uv_env = os.environ.get("UV_PROJECT_ENVIRONMENT") or check_uv_available()
    if is_uv_env:
        print("✅ Running with uv environment")
    
    # Configuration
    host = "0.0.0.0"
    port = 8000
    reload = "--reload" in sys.argv or "-r" in sys.argv
    
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"API Documentation: http://localhost:{port}/docs")
    print(f"Alternative Docs: http://localhost:{port}/redoc")
    print("=" * 50)
    
    # Start the server
    try:
        uvicorn.run(
            "main:app", 
            host=host, 
            port=port, 
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
