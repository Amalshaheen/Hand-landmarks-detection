#!/usr/bin/env python3
"""
Helper script to download the MediaPipe hand landmarker model.
"""

import os
import sys
import urllib.request


def download_model():
    """Download the hand landmarker model."""
    model_path = 'hand_landmarker.task'
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"Model file already exists: {model_path}")
        print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        if file_size < 1000000:  # Less than 1MB probably means it's corrupted
            print("Warning: File size seems too small. It might be corrupted.")
            response = input("Do you want to re-download? (y/n): ")
            if response.lower() != 'y':
                return
            os.remove(model_path)
        else:
            return
    
    print(f"Downloading hand landmarker model...")
    print(f"URL: {url}")
    print(f"Destination: {model_path}")
    print()
    
    try:
        # Try with urllib first
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(block_num * block_size / total_size * 100, 100)
                downloaded = min(block_num * block_size, total_size)
                print(f"\rProgress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end='')
        
        urllib.request.urlretrieve(url, model_path, reporthook)
        print()  # New line after progress
        
        # Verify download
        file_size = os.path.getsize(model_path)
        print(f"\n✓ Model downloaded successfully!")
        print(f"  File: {model_path}")
        print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        if file_size < 1000000:
            print("\nWarning: Downloaded file seems too small. Download may have failed.")
            print("Please try downloading manually from:")
            print(url)
            return False
            
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nPlease download the model manually:")
        print(f"1. Visit: {url}")
        print(f"2. Save the file as '{model_path}' in this directory")
        print(f"   Current directory: {os.getcwd()}")
        print("\nAlternatively, try using wget or curl:")
        print(f"   wget {url}")
        print(f"   or")
        print(f"   curl -L -o {model_path} {url}")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
