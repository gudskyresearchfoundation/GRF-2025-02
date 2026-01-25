"""
Precision Agriculture System
Entry Point - Launches FastAPI Server

This file starts the server.py using uvicorn
"""

import os
import sys
import subprocess

def main():
    """Launch the FastAPI server with uvicorn"""
    
    print("=" * 70)
    print("🌱 PRECISION AGRICULTURE SYSTEM")
    print("=" * 70)
    print("🚀 Starting server...")
    print("📍 Server will run at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 70)
    print()
    
    try:
        # Launch server.py using uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"  # Auto-reload on code changes
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check if port 8000 is already in use")
        print("3. Ensure you're in the project root directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
