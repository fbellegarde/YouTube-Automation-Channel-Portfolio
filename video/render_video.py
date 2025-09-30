import os
import logging
import re
import json
import random
import time
from typing import Optional, List, Dict, Any
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from gtts import gTTS
import pandas as pd
import moviepy.editor as mp
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from etl.image_downloader import search_and_download_images, create_placeholder, cleanup_downloaded_images

# --- CONFIGURATION ---
load_dotenv()
IMAGE_MAGICK_PATH = os.getenv("IMAGE_MAGICK_PATH", r'C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe')
mp.ImageMagick_binary = IMAGE_MAGICK_PATH

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/render.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Client: {e}")
        genai = None
else:
    logger.error("GEMINI_API_KEY not found. LLM script generation will fail.")
    genai = None

# --- DATABASE DEFINITION ---
Base = declarative_base()

class Show(Base):
    __tablename__ = 'shows'
    id = Column(Integer, primary_key=True)
    title = Column(String, unique=True)
    genres = Column(Text)
    summary = Column(Text)
    actors = Column(Text)
    year = Column(String)
    imdb_rating = Column(String)
    network = Column(String)
    premiered = Column(String)
    status = Column(String)
    image_url = Column(String)

    def __repr__(self):
        return f"<Show(title='{self.title}')>"

# --- LLM SCRIPT GENERATION FUNCTION ---
def generate_llm_script(title: str, summary: str, actors: str) -> Optional[str]:
    """Generates a long-form YouTube script using the Gemini API."""
    if not genai:
        logger.error("Gemini client is not initialized.")
        return None

    prompt = f"""
    Write a detailed, 900-word, 5-section YouTube video script about the classic TV show: "{title}".
    
    Data Points to Include:
    - Primary Summary: "{summary}"
    - Key Cast/Actors: {actors}
    
    Structure the script clearly with headings and a professional, engaging tone suitable for narration. The script must be high quality and must exceed 850 words.
    
    Sections should cover:
    1. Introduction and Origin Story.
    2. Deep Dive into Main Characters and Voice Acting.
    3. Cultural Impact, Fan Theories, and Trivia.
    4. Production Challenges, Successes, and Awards.
    5. Final Legacy and Modern Relevance.
    """
    
    try:
        response = genai.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7
            }
        )
        script_text = re.sub(r'#+\s*', '', response.text).strip()
        word_count = len(script_text.split())
        
        if word_count < 850:
            logger.warning(f"LLM script only generated {word_count} words. Using it, but video may be slightly short.")
        else:
            logger.info(f"LLM successfully generated script with {word_count} words.")

        return script_text
        
    except Exception as e:
        logger.error(f"LLM Error for {title}: {e}")
        return None

# --- TTS FUNCTION ---
def generate_pro_audio(text: str, output_path: str) -> Optional[float]:
    """TTS using gTTS. Requires a long script to meet duration."""
    MIN_DURATION_SECONDS = 240  # 4 minutes minimum
    
    try:
        word_count = len(text.split())
        if word_count < 700:
            logger.warning(f"Script only has {word_count} words. Expect short video.")
            
        tts = gTTS(text=text, lang='en')
        tts.save(output_path)
        
        audio_clip = AudioFileClip(output_path)
        true_duration = audio_clip.duration
        audio_clip.close()
        
        if true_duration < MIN_DURATION_SECONDS:
            logger.error(f"TTS result duration ({true_duration:.2f}s) is too short. Need >= {MIN_DURATION_SECONDS}s. Aborting.")
            return None
             
        return true_duration
        
    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return None

# --- DYNAMIC SLIDESHOW WITH KEN BURNS EFFECT ---
def apply_ken_burns(t: float, clip: ImageClip, clip_duration: float, start_scale: float, end_scale: float, start_center: tuple, end_center: tuple, size: tuple = (1280, 720)) -> np.ndarray:
    """Apply Ken Burns effect to a frame at time t."""
    W, H = size
    ratio = min(1.0, t / clip_duration)
    
    current_scale = start_scale + (end_scale - start_scale) * ratio
    current_center_x = start_center[0] + (end_center[0] - start_center[0]) * ratio
    current_center_y = start_center[1] + (end_center[1] - start_center[1]) * ratio
    
    # Get the frame at time t
    frame = clip.get_frame(t)
    
    # Ensure frame is RGB
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    
    pil_img = Image.fromarray(frame)
    
    zoomed_width = int(W * current_scale)
    zoomed_height = int(H * current_scale)
    
    zoomed_img = pil_img.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
    
    target_W, target_H = W, H
    x_start = int(current_center_x * zoomed_img.width - target_W / 2)
    y_start = int(current_center_y * zoomed_img.height - target_H / 2)

    x_start = max(0, min(x_start, zoomed_img.width - target_W))
    y_start = max(0, min(y_start, zoomed_img.height - target_H))
    
    cropped_img = zoomed_img.crop((x_start, y_start, x_start + target_W, y_start + target_H))
    
    return np.array(cropped_img).astype(np.uint8)

def create_dynamic_slideshow(image_paths: List[str], duration: float, size: tuple = (1280, 720)) -> CompositeVideoClip:
    """Creates a dynamic video clip cycling through images with Ken Burns effect."""
    if not image_paths:
        placeholder_path = create_placeholder("data/placeholder.jpg", "No Images")
        return ImageClip(placeholder_path, duration=duration).set_position('center').set_opacity(0.8).resize(size)

    num_images = len(image_paths)
    clip_duration = duration / num_images
    final_clips = []
    
    for path in image_paths:
        try:
            base_clip = ImageClip(path, duration=clip_duration)
            
            start_scale = random.uniform(1.0, 1.05)
            end_scale = random.uniform(1.05, 1.10) if random.random() < 0.5 else random.uniform(0.95, 1.0)
            if end_scale < 1.0:
                end_scale = 1.05

            start_center = (random.uniform(0.3, 0.7), random.uniform(0.3, 0.7))
            end_center = (random.uniform(0.3, 0.7), random.uniform(0.3, 0.7))
            
            final_clip = base_clip.fl(
                lambda t: apply_ken_burns(t, base_clip, clip_duration, start_scale, end_scale, start_center, end_center, size),
                apply_to=['video']
            ).set_duration(clip_duration).resize(size).set_opacity(0.8)
            
            final_clips.append(final_clip)
        except Exception as e:
            logger.error(f"Error processing image {path}: {e}")
            placeholder_path = create_placeholder("data/placeholder.jpg", "No Images")
            final_clips.append(ImageClip(placeholder_path, duration=clip_duration).resize(size).set_opacity(0.8))
    
    return concatenate_videoclips(final_clips, method="compose").set_duration(duration)

# --- HELPER FUNCTIONS ---
def create_file_safe_name(title: str) -> str:
    """Creates a standardized file name from a title."""
    safe_title = title.replace("'", "").replace(" ", "_")
    return re.sub(r'[^\w-]', '', safe_title)

def parse_db_field(data: str) -> list:
    """Safely parses a JSON list string from the database."""
    if pd.notnull(data) and data != 'N/A' and data.strip():
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON data: {data[:50]}...")
    return []

# --- MASTER VIDEO CREATION FUNCTION ---
def create_video(row: Dict[str, Any]):
    """Orchestrates script generation, media download, and video rendering."""
    title = row['title']
    safe_title = create_file_safe_name(title)
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    os.makedirs('scripts', exist_ok=True)
    
    narration_file = f"data/{safe_title}_narration.mp3"
    video_file = f"videos/{safe_title}.mp4"
    NUM_SLIDES = 8

    # Skip if video exists
    if os.path.exists(video_file):
        logger.info(f"Video already exists for {title}: {video_file}")
        return

    # --- 1. Script Generation (LLM Integration) ---
    script_file = f"scripts/{safe_title}_script.txt"
    narration_text = ""
    
    if os.path.exists(script_file):
        with open(script_file, 'r', encoding='utf-8') as f:
            narration_text = f.read()
            
    if len(narration_text.split()) < 600:
        logger.info(f"Script for {title} is missing or too short. Generating with LLM.")
        actors_list = parse_db_field(row.get('actors', '[]'))
        top_actors = ', '.join([a.strip() for a in actors_list if isinstance(a, str)][:8])
        
        narration_text = generate_llm_script(
            title=title,
            summary=row.get('summary', 'a classic animated series'),
            actors=top_actors
        )
        
        if narration_text:
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(narration_text)
        else:
            logger.error(f"Failed to generate LLM script for {title}. Using summary.")
            narration_text = row.get('summary', f"{title} is a beloved animated series!")

    # --- 2. Audio Generation and Timing ---
    duration = generate_pro_audio(narration_text, narration_file)
    
    if not duration:
        logger.error(f"Audio generation failed or duration was too short for {title}. Aborting.")
        return

    # --- 3. Image Sourcing (Pexels) ---
    image_paths = search_and_download_images(title, num_images=NUM_SLIDES)
    if not image_paths:
        logger.error(f"Image download failed for {title}. Proceeding with placeholder.")
        image_paths = [create_placeholder(f"data/{safe_title}_placeholder.jpg", title)]

    # --- 4. Video Composition ---
    try:
        audio_clip = AudioFileClip(narration_file)
        slideshow_clip = create_dynamic_slideshow(image_paths, duration)
        video_clip = slideshow_clip.set_audio(audio_clip)
        
        txt_clip = TextClip(
            f"{title}: The Full Story",
            fontsize=70, color='yellow', bg_color='transparent', font='Arial-Bold',
            stroke_color='black', stroke_width=2
        ).set_position(('center', 50)).set_duration(duration)
        
        final_clip = CompositeVideoClip([video_clip, txt_clip], size=(1280, 720))
        
        final_clip.write_videofile(
            video_file, fps=24, codec='mpeg4', audio_codec='mp3', bitrate='8000k',
            temp_audiofile='temp-audio.mp3', remove_temp=True, logger=None
        )
        logger.info(f"Created DYNAMIC video: {video_file} with duration {duration:.2f}s")
    except Exception as e:
        logger.error(f"FATAL MOVIEPY WRITE ERROR for {title}: {e}")
    finally:
        if os.path.exists(narration_file):
            try:
                os.remove(narration_file)
                logger.info(f"Removed narration file: {narration_file}")
            except:
                logger.warning(f"Failed to remove {narration_file}")
        cleanup_downloaded_images(safe_title, NUM_SLIDES)
        logger.info(f"Cleanup complete for {title}.")

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    logger.info("Starting video rendering process...")
    
    try:
        engine = create_engine('sqlite:///db/local.db')
        Base.metadata.bind = engine
        Session = sessionmaker(bind=engine)
        session = Session()
        
        shows_to_render = session.query(Show).all()
        
        if not shows_to_render:
            logger.error("No show data found in 'db/local.db'. Please run 'python -m etl.extract_data' first.")
            session.close()
            exit(1)

        logger.info(f"Found {len(shows_to_render)} shows to render.")
        
        for show in shows_to_render:
            row_data = {
                'title': show.title,
                'summary': show.summary,
                'year': show.year,
                'actors': show.actors
            }
            logger.info(f"--- Starting video creation for: {show.title} ---")
            create_video(row_data)
            logger.info(f"--- Finished video creation for: {show.title} ---\n")
            time.sleep(2)
            
    except Exception as e:
        logger.error(f"A fatal error occurred during database access or rendering: {e}")
    finally:
        if 'session' in locals() and session:
            session.close()

    logger.info("Video rendering process completed.")