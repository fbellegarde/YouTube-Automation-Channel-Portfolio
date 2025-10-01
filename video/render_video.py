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
from pydub import AudioSegment
import subprocess

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
        logging.FileHandler('logs/render.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Gemini client initialized with gemini-2.5-flash.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Client: {e}")
        model = None
else:
    logger.error("GEMINI_API_KEY not found. LLM script generation will use fallback.")
    model = None

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
def generate_llm_script(title: str, summary: str, actors: str, genres: str, network: str, premiered: str, imdb_rating: str) -> Optional[str]:
    """Generates a 900-1200 word YouTube script using Gemini API."""
    if not model:
        logger.error("Gemini model is not initialized.")
        return None

    # Sanitize inputs to avoid safety filter issues
    summary = re.sub(r'[^\w\s.,!?]', '', summary)  # Remove special characters
    actors = re.sub(r'[^\w\s,]', '', actors)
    genres = re.sub(r'[^\w\s,]', '', genres)

    prompt = f"""
    Create a 900-1200 word YouTube video script for a 4-5 minute video about the TV show "{title}". The script must be engaging, family-friendly, and avoid any copyrighted material or sensitive topics (e.g., violence, politics, adult themes). Use a dynamic, conversational tone suitable for narration.

    Include these details:
    - Summary: "{summary}"
    - Key Cast/Actors: {actors}
    - Genres: {genres}
    - Network: {network}
    - Premiered: {premiered}
    - IMDb Rating: {imdb_rating}

    Structure the script with 5 sections:
    1. Introduction: Hook the viewer, introduce the show, mention premiere year and network.
    2. Origin Story: Describe the show's creation, creators, and inspirations.
    3. Characters and Cast: Highlight main characters, their arcs, and notable actors.
    4. Cultural Impact: Share fan reactions, iconic moments, and trivia.
    5. Legacy: Discuss awards, influence, and why the show matters today.

    Use varied sentence structures, energetic language, and no repetition. Ensure the script is safe for all audiences and complies with YouTube's community guidelines.
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 1500
            }
        )
        script_text = re.sub(r'#+\s*', '', response.text).strip()
        word_count = len(script_text.split())
        
        logger.info(f"LLM script for {title}: {word_count} words generated.")
        if word_count < 900:
            logger.warning(f"LLM script for {title} is {word_count} words, below target. Extending in fallback.")
            return None  # Fallback to generate_fallback_script
        elif word_count > 1200:
            logger.info(f"Trimming LLM script for {title} from {word_count} to 1200 words.")
            script_text = ' '.join(script_text.split()[:1200])
        return script_text
        
    except Exception as e:
        logger.error(f"LLM Error for {title}: {e}")
        return None

# --- TTS FUNCTION WITH AUDIO ENHANCEMENT ---
def generate_pro_audio(text: str, output_path: str) -> Optional[float]:
    """Generates TTS audio with pacing and background music."""
    MIN_DURATION_SECONDS = 240  # 4 minutes
    MAX_DURATION_SECONDS = 300  # 5 minutes
    
    try:
        word_count = len(text.split())
        logger.info(f"Generating audio for {output_path} with {word_count} words.")
        
        # Generate base TTS
        tts = gTTS(text=text, lang='en', tld='com', slow=False)
        temp_path = output_path.replace('.mp3', '_temp.mp3')
        tts.save(temp_path)
        
        # Load with pydub for enhancements
        audio = AudioSegment.from_mp3(temp_path)
        duration_ms = len(audio)
        
        if duration_ms == 0:
            logger.error(f"Empty audio generated for {output_path}. Aborting.")
            os.remove(temp_path)
            return None
        
        # Estimate words per minute (WPM) for initial speed
        estimated_wpm = (word_count / (duration_ms / 1000)) * 60
        logger.info(f"Initial audio duration: {duration_ms/1000:.2f}s, estimated WPM: {estimated_wpm:.2f}")
        
        # Target 180-200 WPM for 4-5 min video
        target_wpm = 190
        target_duration = (word_count / target_wpm) * 60 * 1000  # Target duration in ms
        speed_factor = duration_ms / target_duration if duration_ms != 0 else 1.0
        speed_factor = max(0.8, min(1.3, speed_factor))  # Clamp 0.8x-1.3x
        
        audio = audio.speedup(playback_speed=speed_factor, crossfade=30)
        duration_ms = len(audio)
        
        # Add background music
        music_path = 'data/upbeat_background.mp3'
        if os.path.exists(music_path):
            try:
                # Verify MP3
                ffprobe_cmd = f'ffmpeg -i "{music_path}" -f null -'
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, shell=True)
                if result.returncode != 0:
                    logger.error(f"Invalid MP3 file at {music_path}: {result.stderr}")
                    os.remove(temp_path)
                    return None
                
                music = AudioSegment.from_mp3(music_path)
                while len(music) < len(audio):
                    music += music
                music = music[:len(audio)] - 20  # Fade out, 20dB quieter
                audio = audio.overlay(music, position=0)
            except Exception as e:
                logger.error(f"Failed to process background music for {output_path}: {e}")
                # Proceed without music
        else:
            logger.warning(f"Background music not found at {music_path}. Proceeding without music.")
        
        audio.export(output_path, format='mp3')
        duration = len(audio) / 1000.0
        
        # Final duration check
        if duration < MIN_DURATION_SECONDS or duration > MAX_DURATION_SECONDS:
            logger.warning(f"TTS duration ({duration:.2f}s) outside 4-5 min range for {output_path}. Adjusting.")
            target_duration = 270  # Aim for middle of 240-300s
            speed_factor = duration_ms / (target_duration * 1000)
            speed_factor = max(0.8, min(1.3, speed_factor))
            audio = AudioSegment.from_mp3(temp_path).speedup(playback_speed=speed_factor, crossfade=30)
            if os.path.exists(music_path):
                music = AudioSegment.from_mp3(music_path)
                while len(music) < len(audio):
                    music += music
                music = music[:len(audio)] - 20
                audio = audio.overlay(music, position=0)
            audio.export(output_path, format='mp3')
            duration = len(audio) / 1000.0
        
        logger.info(f"Generated audio for {output_path} with duration {duration:.2f}s")
        os.remove(temp_path)
        return duration
        
    except Exception as e:
        logger.error(f"TTS generation error for {output_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

# --- DYNAMIC SLIDESHOW WITH TRANSITIONS ---
def apply_ken_burns(clip: ImageClip, t: float, clip_duration: float, start_scale: float, end_scale: float, start_center: tuple, end_center: tuple, size: tuple = (1280, 720)) -> np.ndarray:
    """Apply Ken Burns effect to a frame at time t."""
    W, H = size
    ratio = min(1.0, t / clip_duration)
    
    current_scale = start_scale + (end_scale - start_scale) * ratio
    current_center_x = start_center[0] + (end_center[0] - start_center[0]) * ratio
    current_center_y = start_center[1] + (end_center[1] - start_center[1]) * ratio
    
    frame = clip.get_frame(t)
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

def create_dynamic_slideshow(image_paths: List[str], duration: float, title: str, size: tuple = (1280, 720)) -> CompositeVideoClip:
    """Creates a dynamic slideshow with crossfade transitions and text overlays."""
    if not image_paths:
        placeholder_path = create_placeholder("data/placeholder.jpg", "No Images")
        return ImageClip(placeholder_path, duration=duration).set_position('center').resize(size)

    num_images = len(image_paths)
    clip_duration = duration / num_images
    final_clips = []
    
    for i, path in enumerate(image_paths):
        try:
            base_clip = ImageClip(path, duration=clip_duration)
            start_scale = random.uniform(1.0, 1.05)
            end_scale = random.uniform(1.05, 1.10)
            start_center = (random.uniform(0.3, 0.7), random.uniform(0.3, 0.7))
            end_center = (random.uniform(0.3, 0.7), random.uniform(0.3, 0.7))
            
            ken_burns_clip = base_clip.fl(
                lambda t: apply_ken_burns(base_clip, t, clip_duration, start_scale, end_scale, start_center, end_center, size),
                apply_to=['video']
            ).set_duration(clip_duration).resize(size).set_opacity(0.9)
            
            if i > 0:
                ken_burns_clip = ken_burns_clip.crossfadein(0.5)
            
            text = f"Key Moment: {title} Scene {i+1}"
            txt_clip = TextClip(
                text, fontsize=50, color='white', bg_color='transparent',
                font='Arial-Bold', stroke_color='black', stroke_width=1
            ).set_position(('center', 'bottom')).set_duration(clip_duration).set_opacity(0.8)
            
            final_clip = CompositeVideoClip([ken_burns_clip, txt_clip], size=size)
            final_clips.append(final_clip)
            base_clip.close()  # Free memory
            ken_burns_clip.close()
            txt_clip.close()
        except Exception as e:
            logger.error(f"Error processing image {path}: {e}")
            placeholder_path = create_placeholder(f"data/{title.replace(' ', '_')}_placeholder.jpg", title)
            final_clips.append(ImageClip(placeholder_path, duration=clip_duration).resize(size))
    
    slideshow = concatenate_videoclips(final_clips, method="compose").set_duration(duration)
    final_clip = CompositeVideoClip([slideshow], size=size)
    for clip in final_clips:
        clip.close()
    return final_clip

# --- HELPER FUNCTIONS ---
def create_file_safe_name(title: str) -> str:
    """Creates a standardized file name, preserving apostrophes and exclamation marks."""
    safe_title = title.replace(" ", "_")
    safe_title = re.sub(r'[^\w\'!_-]', '', safe_title)
    return safe_title

def parse_db_field(data: str) -> list:
    """Parses a database field, handling both JSON and comma-separated text."""
    if pd.notnull(data) and data != 'N/A' and data.strip():
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON data: {data[:50]}... Treating as comma-separated.")
            return [item.strip() for item in data.split(',') if item.strip()]
    return []

def generate_fallback_script(title: str, summary: str, actors: str, genres: str, network: str, premiered: str, imdb_rating: str) -> str:
    """Generates a ~900-1200 word fallback script when LLM fails."""
    actors_list = parse_db_field(actors)
    genres_list = parse_db_field(genres)
    top_actors = ', '.join([a.strip() for a in actors_list if isinstance(a, str)][:8])
    genres_text = ', '.join([g.strip() for g in genres_list if isinstance(g, str)])
    
    # Base script (~600-700 words)
    script = f"""
    Welcome to our deep dive into {title}! Premiered in {premiered}, this {genres_text} gem aired on {network} with a {imdb_rating} IMDb rating. Featuring {top_actors}, here's a quick overview: {summary}. Let's uncover why this show became a fan favorite!

    **The Show's Origin**: {title} emerged with a bold vision, blending {genres_text} to captivate audiences. Launched in {premiered} on {network}, it drew from pop culture and creative storytelling. The creators crafted episodes that resonated with both kids and adults, using clever humor and heartfelt moments. Its {imdb_rating} rating shows its widespread appeal, making it a standout in its time.

    **Characters That Shine**: The heart of {title} is its vibrant characters. {top_actors} delivered unforgettable performances, bringing depth to each role. {summary} Their unique personalities drove the narrative, creating moments fans still cherish. From action-packed scenes to comedic highlights, the cast infused every episode with energy.

    **Cultural Impact**: {title} left a lasting mark on pop culture. Its catchphrases and memes became iconic, with fans loving the clever references to films and real-world events. Fun fact: many episodes were inspired by the creators' own experiences, adding authenticity. This kept viewers hooked, contributing to its {imdb_rating} rating.

    **Why It Still Matters**: Years after its {premiered} debut, {title} remains timeless. Its themes of friendship, adventure, and {genres_text} resonate today. The show earned awards and inspired fan art, spin-offs, and more. Its influence lives on in modern series, proving its enduring legacy.

    **Fun Facts and Trivia**: Fans adore {title}'s quirky details. From behind-the-scenes stories to fan theories, there's always something new to discover. {top_actors} often shared their love for the show in interviews. {network}'s bold move to air this {genres_text} series paid off with a {imdb_rating} rating.

    **Behind the Scenes**: Producing {title} was a labor of love. Writers and animators balanced {genres_text} to create meaningful episodes. {top_actors} improvised lines, adding authenticity. {network}'s support let the show push creative boundaries, boosting its {imdb_rating} success.

    **Fan Community**: The {title} fanbase is passionate! Social media buzzes with discussions about favorite episodes and theories. Its {genres_text} appeal drew diverse viewers, from kids to adults. {network}'s {premiered} launch sparked a cultural phenomenon, solidified by its {imdb_rating} rating.
    """
    
    word_count = len(script.split())
    # Extend to 900-1200 words
    while word_count < 900:
        script += f"""
        **Exploring {title}'s Impact**: The show's {genres_text} style evolved over time, with {top_actors} delivering standout performances. Its {premiered} debut on {network} set a new standard for {genres_text} shows, earning a {imdb_rating} rating. Fans still celebrate its clever writing and memorable moments, keeping the community alive.
        """
        word_count = len(script.split())
    
    if word_count < 900:
        logger.warning(f"Fallback script for {title} has {word_count} words. May be slightly short.")
    elif word_count > 1200:
        logger.info(f"Trimming fallback script for {title} from {word_count} to 1200 words.")
        script = ' '.join(script.split()[:1200])
    
    logger.info(f"Fallback script for {title}: {word_count} words generated.")
    return script

# --- MASTER VIDEO CREATION FUNCTION ---
def create_video(row: Dict[str, Any]) -> bool:
    """Orchestrates script generation, media download, and video rendering."""
    title = row['title']
    safe_title = create_file_safe_name(title)
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    os.makedirs('scripts', exist_ok=True)
    
    narration_file = f"data/{safe_title}_narration.mp3"
    video_file = f"videos/{safe_title}.mp4"
    NUM_SLIDES = 12

    if os.path.exists(video_file):
        logger.info(f"Video already exists for {title}: {video_file}")
        return True

    # --- 1. Script Generation ---
    script_file = f"scripts/{safe_title}_script.txt"
    narration_text = ""
    
    if os.path.exists(script_file):
        with open(script_file, 'r', encoding='utf-8') as f:
            narration_text = f.read()
            
    if len(narration_text.split()) < 900:
        logger.info(f"Script for {title} is missing or too short. Generating with LLM.")
        actors_list = parse_db_field(row.get('actors', ''))
        genres_list = parse_db_field(row.get('genres', ''))
        top_actors = ', '.join([a.strip() for a in actors_list if isinstance(a, str)][:8])
        genres_text = ', '.join([g.strip() for g in genres_list if isinstance(g, str)])
        
        narration_text = generate_llm_script(
            title=title,
            summary=row.get('summary', 'A classic animated series'),
            actors=top_actors,
            genres=genres_text,
            network=row.get('network', 'N/A'),
            premiered=row.get('premiered', 'N/A'),
            imdb_rating=row.get('imdb_rating', 'N/A')
        )
        
        if not narration_text:
            logger.error(f"Failed to generate LLM script for {title}. Using enhanced fallback.")
            narration_text = generate_fallback_script(
                title=title,
                summary=row.get('summary', 'A classic animated series'),
                actors=top_actors,
                genres=genres_text,
                network=row.get('network', 'N/A'),
                premiered=row.get('premiered', 'N/A'),
                imdb_rating=row.get('imdb_rating', 'N/A')
            )
        
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(narration_text)

    # --- 2. Audio Generation ---
    duration = generate_pro_audio(narration_text, narration_file)
    
    if not duration:
        logger.error(f"Audio generation failed for {title}. Continuing to next show.")
        return False

    # --- 3. Image Sourcing ---
    image_paths = search_and_download_images(title, num_images=NUM_SLIDES)
    if not image_paths:
        logger.error(f"Image download failed for {title}. Proceeding with placeholder.")
        image_paths = [create_placeholder(f"data/{safe_title}_placeholder.jpg", title)]

    # --- 4. Video Composition ---
    try:
        audio_clip = AudioFileClip(narration_file)
        slideshow_clip = create_dynamic_slideshow(image_paths, duration, title)
        video_clip = slideshow_clip.set_audio(audio_clip)
        
        title_clip = TextClip(
            f"{title}: The Full Story",
            fontsize=70, color='yellow', bg_color='transparent', font='Arial-Bold',
            stroke_color='black', stroke_width=2
        ).set_position(('center', 50)).set_duration(duration)
        
        final_clip = CompositeVideoClip([video_clip, title_clip], size=(1280, 720))
        
        final_clip.write_videofile(
            video_file, fps=24, codec='mpeg4', audio_codec='mp3', bitrate='8000k',
            temp_audiofile='temp-audio.mp3', remove_temp=True, logger=None
        )
        logger.info(f"Created DYNAMIC video: {video_file} with duration {duration:.2f}s")
        audio_clip.close()
        slideshow_clip.close()
        video_clip.close()
        title_clip.close()
        final_clip.close()
        return True
    except Exception as e:
        logger.error(f"FATAL MOVIEPY WRITE ERROR for {title}: {e}")
        return False
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
def main():
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
                'actors': show.actors,
                'genres': show.genres,
                'network': show.network,
                'premiered': show.premiered,
                'imdb_rating': show.imdb_rating
            }
            logger.info(f"--- Starting video creation for: {show.title} ---")
            success = create_video(row_data)
            if success:
                logger.info(f"--- Successfully created video for: {show.title} ---")
            else:
                logger.warning(f"--- Failed to create video for: {show.title}, continuing to next show ---")
            time.sleep(2)
            
    except Exception as e:
        logger.error(f"A fatal error occurred during database access or rendering: {e}")
    finally:
        if 'session' in locals() and session:
            session.close()

    logger.info("Video rendering process completed.")

if __name__ == '__main__':
    main()