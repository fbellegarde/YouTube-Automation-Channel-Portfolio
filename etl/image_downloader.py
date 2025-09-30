import os
import logging
import json
from typing import List
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Configure logger (assuming it's set up in render_video.py as well)
logger = logging.getLogger(__name__)

# --- PEXELS CONFIGURATION ---
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_BASE_URL = "https://api.pexels.com/v1/search"

# =================================================================
# 💡 UPGRADE 1: PEXELS API SEARCH
# =================================================================

def search_and_download_images(title: str, num_images: int = 6) -> List[str]:
    """
    Searches the Pexels API for commercially free-to-use images and downloads them.
    All images are licensed under the Pexels License (free for commercial use, no attribution needed).
    """
    if not PEXELS_API_KEY:
        logger.error("FATAL: PEXELS_API_KEY not found. Aborting image download.")
        return []

    safe_title = title.replace("'", "").replace(" ", "_")
    downloaded_paths = []
    
    # Define search query - ensuring a good thematic match
    query = f"{title} characters"

    headers = {
        'Authorization': PEXELS_API_KEY
    }
    params = {
        'query': query,
        'per_page': num_images * 2,  # Request more than needed
        'orientation': 'landscape',
        'size': 'large'              # Request high-quality images
    }

    try:
        # 1. Execute Search
        response = requests.get(PEXELS_BASE_URL, headers=headers, params=params, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        photos = data.get('photos', [])
        logger.info(f"Pexels found {len(photos)} results for '{title}'.")

        # 2. Iterate and Download
        for photo in photos:
            if len(downloaded_paths) >= num_images:
                break
            
            # Use 'large' or 'medium' size URLs for downloading
            image_url = photo['src'].get('large') or photo['src'].get('medium')

            if image_url:
                file_path = os.path.join('data', f"{safe_title}_img_{len(downloaded_paths)}.jpg")
                
                try:
                    # Download the image data
                    img_response = requests.get(image_url, timeout=10)
                    img_response.raise_for_status()

                    # Process image with PIL (Pillow)
                    img = Image.open(BytesIO(img_response.content)).convert('RGB')
                    
                    # 💡 Expert Media Processing: Resize and Pad to 16:9 (1280x720)
                    W, H = 1280, 720
                    # This line correctly uses modern PIL resampling constant
                    img.thumbnail((W, H), Image.Resampling.LANCZOS) # Resize preserving aspect ratio
                    
                    final_img = Image.new('RGB', (W, H), color='black') # Create black canvas
                    x_offset = (W - img.width) // 2
                    y_offset = (H - img.height) // 2
                    final_img.paste(img, (x_offset, y_offset)) # Center the image
                    
                    final_img.save(file_path, 'JPEG', quality=90)
                    downloaded_paths.append(file_path)
                    logger.info(f"Downloaded and processed image {len(downloaded_paths)} for {title} to {file_path}")

                except Exception as dl_e:
                    logger.warning(f"Failed to process image from {image_url}: {dl_e}")
                    
    except requests.exceptions.RequestException as api_e:
        logger.error(f"Pexels API search error for {title}: {api_e}")
        
    return downloaded_paths

# --- Placeholder function for safety (required in render_video.py) ---
def create_placeholder(path: str, title: str) -> str:
    """Creates a basic placeholder image if API fails."""
    placeholder_dir = 'data'
    os.makedirs(placeholder_dir, exist_ok=True)
    
    placeholder_path = os.path.join(placeholder_dir, 'placeholder.jpg')

    # Simple check to avoid recreating the file if it exists
    if os.path.exists(placeholder_path):
        return placeholder_path
    
    # Create a simple black image with text
    try:
        img = Image.new('RGB', (1280, 720), color = (0, 0, 0))
        # Note: You can add text with ImageDraw if you have fonts available, 
        # but leaving it simple to avoid FontFileNotFound errors.
        img.save(placeholder_path, 'JPEG')
        logger.info(f"Created generic placeholder at {placeholder_path}")
        return placeholder_path
    except Exception as e:
        logger.error(f"Failed to create placeholder image: {e}")
        return path # Return the intended path even if file creation failed

# =================================================================
# 💡 CRITICAL: Clean-up function for all downloaded images
# =================================================================
def cleanup_downloaded_images(safe_title: str, num_images: int):
    """Removes the dynamically generated image files after video creation."""
    for i in range(num_images):
        indexed_path = os.path.join('data', f"{safe_title}_img_{i}.jpg")
        if os.path.exists(indexed_path):
            os.remove(indexed_path)
            # logger.info(f"Removed temporary image file: {indexed_path}")