import os
import logging
import json
from typing import List
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()

# --- PEXELS AND PIXABAY CONFIGURATION ---
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
PEXELS_BASE_URL = "https://api.pexels.com/v1/search"
PIXABAY_BASE_URL = "https://pixabay.com/api/"

def search_and_download_images(title: str, num_images: int = 12) -> List[str]:
    """
    Searches Pexels and Pixabay for copyright-free images (Pexels License, Pixabay License).
    Downloads and processes images to 1280x720.
    """
    if not PEXELS_API_KEY and not PIXABAY_API_KEY:
        logger.error("FATAL: Neither PEXELS_API_KEY nor PIXABAY_API_KEY found. Aborting image download.")
        return []

    safe_title = title.replace("'", "").replace(" ", "_")
    downloaded_paths = []
    
    # Try Pexels first
    if PEXELS_API_KEY:
        query = f"{title} characters"
        headers = {'Authorization': PEXELS_API_KEY}
        params = {
            'query': query,
            'per_page': num_images * 2,
            'orientation': 'landscape',
            'size': 'large'
        }
        
        try:
            response = requests.get(PEXELS_BASE_URL, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            photos = data.get('photos', [])
            logger.info(f"Pexels found {len(photos)} results for '{title}'.")
            
            for photo in photos:
                if len(downloaded_paths) >= num_images:
                    break
                image_url = photo['src'].get('large') or photo['src'].get('medium')
                if image_url:
                    file_path = os.path.join('data', f"{safe_title}_img_{len(downloaded_paths)}.jpg")
                    try:
                        img_response = requests.get(image_url, timeout=10)
                        img_response.raise_for_status()
                        img = Image.open(BytesIO(img_response.content)).convert('RGB')
                        W, H = 1280, 720
                        img.thumbnail((W, H), Image.Resampling.LANCZOS)
                        final_img = Image.new('RGB', (W, H), color='black')
                        x_offset = (W - img.width) // 2
                        y_offset = (H - img.height) // 2
                        final_img.paste(img, (x_offset, y_offset))
                        final_img.save(file_path, 'JPEG', quality=90)
                        downloaded_paths.append(file_path)
                        logger.info(f"Downloaded Pexels image {len(downloaded_paths)} for {title} to {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to process Pexels image from {image_url}: {e}")
        except Exception as e:
            logger.error(f"Pexels API error for {title}: {e}")

    # Fallback to Pixabay if needed
    if len(downloaded_paths) < num_images and PIXABAY_API_KEY:
        remaining = num_images - len(downloaded_paths)
        params = {
            'key': PIXABAY_API_KEY,
            'q': f"{title} characters",
            'image_type': 'photo',
            'per_page': remaining * 2,
            'orientation': 'horizontal'
        }
        
        try:
            response = requests.get(PIXABAY_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            photos = data.get('hits', [])
            logger.info(f"Pixabay found {len(photos)} results for '{title}'.")
            
            for photo in photos:
                if len(downloaded_paths) >= num_images:
                    break
                image_url = photo['largeImageURL']
                if image_url:
                    file_path = os.path.join('data', f"{safe_title}_img_{len(downloaded_paths)}.jpg")
                    try:
                        img_response = requests.get(image_url, timeout=10)
                        img_response.raise_for_status()
                        img = Image.open(BytesIO(img_response.content)).convert('RGB')
                        W, H = 1280, 720
                        img.thumbnail((W, H), Image.Resampling.LANCZOS)
                        final_img = Image.new('RGB', (W, H), color='black')
                        x_offset = (W - img.width) // 2
                        y_offset = (H - img.height) // 2
                        final_img.paste(img, (x_offset, y_offset))
                        final_img.save(file_path, 'JPEG', quality=90)
                        downloaded_paths.append(file_path)
                        logger.info(f"Downloaded Pixabay image {len(downloaded_paths)} for {title} to {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to process Pixabay image from {image_url}: {e}")
        except Exception as e:
            logger.error(f"Pixabay API error for {title}: {e}")

    if not downloaded_paths:
        logger.error(f"No images downloaded for {title}. Using placeholder.")
        downloaded_paths = [create_placeholder(f"data/{safe_title}_placeholder.jpg", title)]
    
    return downloaded_paths

def create_placeholder(path: str, title: str) -> str:
    """Creates a placeholder image if API fails."""
    placeholder_dir = 'data'
    os.makedirs(placeholder_dir, exist_ok=True)
    
    placeholder_path = os.path.join(placeholder_dir, f"{title.replace(' ', '_')}_placeholder.jpg")
    
    if os.path.exists(placeholder_path):
        return placeholder_path
    
    try:
        img = Image.new('RGB', (1280, 720), color=(0, 0, 0))
        img.save(placeholder_path, 'JPEG')
        logger.info(f"Created placeholder at {placeholder_path}")
        return placeholder_path
    except Exception as e:
        logger.error(f"Failed to create placeholder image: {e}")
        return path

def cleanup_downloaded_images(safe_title: str, num_images: int):
    """Removes temporary image files."""
    for i in range(num_images):
        indexed_path = os.path.join('data', f"{safe_title}_img_{i}.jpg")
        placeholder_path = os.path.join('data', f"{safe_title}_placeholder.jpg")
        for path in [indexed_path, placeholder_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Removed temporary image: {path}")
                except:
                    logger.warning(f"Failed to remove {path}")