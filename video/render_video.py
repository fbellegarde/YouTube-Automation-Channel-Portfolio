import os
import logging
from typing import Optional
from PIL import Image
from io import BytesIO
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
import imageio
from gtts import gTTS
from sqlalchemy import create_engine
from moviepy.config import change_settings
import requests
import re
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video/render_video.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set FFmpeg and ImageMagick paths
os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\ffmpeg\bin\ffmpeg.exe"
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

def download_image(title: str, url: str, path: str) -> Optional[str]:
    if not url or url == 'N/A':
        logger.warning(f"Invalid image URL for {title}, using placeholder")
        return create_placeholder(path)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'svg' in content_type.lower() or url.lower().endswith('.svg'):
            try:
                from cairosvg import svg2png
                output = BytesIO()
                svg2png(bytestring=response.content, write_to=output)
                img = Image.open(output)
            except Exception as e:
                logger.error(f"SVG conversion error for {title}: {e}")
                return create_placeholder(path)
        else:
            img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img.save(path, 'JPEG', quality=95)
        logger.info(f"Downloaded image for {title} to {path}")
        return path
    except Exception as e:
        logger.error(f"Image download error for {title}: {e}")
        return create_placeholder(path)

def create_placeholder(path: str) -> Optional[str]:
    placeholder = os.path.join('data', 'placeholder.jpg')
    if not os.path.exists(placeholder):
        try:
            img = Image.new('RGB', (1280, 720), color='gray')
            img.save(placeholder)
            logger.info(f"Created placeholder image at {placeholder}")
        except Exception as e:
            logger.error(f"Placeholder creation error: {e}")
            return None
    return placeholder

def create_video(row):
    title = row['title'].replace("'", "").replace(" ", "_")
    narration_file = f"data/{title}_narration.mp3"
    video_file = f"videos/{title}.mp4"
    image_file = f"data/{title}.jpg"
    image_clip = None
    audio_clip = None
    txt_clip = None
    video_clip = None
    final_clip = None
    try:
        # Skip if video already exists
        if os.path.exists(video_file):
            logger.info(f"Video already exists for {row['title']}: {video_file}")
            return
        # Try JSON for Poster URL
        json_file = f"output/{title}.json"
        poster_url = None
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                poster_url = data.get('Poster', 'N/A')
        image_path = download_image(row['title'], poster_url if poster_url and poster_url != 'N/A' else row['image_url'], image_file)
        if not image_path:
            logger.error(f"No image for {row['title']}")
            return
        narration_text = row.get('summary', 'No summary available')
        if not narration_text or len(narration_text.strip()) < 10:
            logger.warning(f"Summary too short for {row['title']}, using default")
            narration_text = f"{row['title']} is a beloved animated series!"
        tts = gTTS(text=narration_text, lang='en')
        tts.save(narration_file)
        logger.info(f"Generated narration to {narration_file}")
        image_clip = ImageClip(image_path).set_duration(10)
        audio_clip = AudioFileClip(narration_file)
        video_clip = image_clip.set_audio(audio_clip)
        txt_clip = TextClip(f"{row['title']}", fontsize=70, color='white', bg_color='black').set_position('center').set_duration(10)
        final_clip = CompositeVideoClip([video_clip, txt_clip])
        final_clip.write_videofile(video_file, fps=24, codec='mpeg4', audio_codec='mp3')
        logger.info(f"Created video: {video_file}")
    except Exception as e:
        logger.error(f"Video creation error for {row['title']}: {e}")
    finally:
        for clip in [image_clip, audio_clip, txt_clip, video_clip, final_clip]:
            if clip and hasattr(clip, 'close'):
                clip.close()
        for file in [narration_file, image_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    logger.info(f"Removed temporary file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file}: {e}")

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    engine = create_engine('sqlite:///db/local.db')
    import pandas as pd
    df = pd.read_sql('SELECT * FROM shows', engine)
    logger.info(f"Found {len(df)} shows in database")
    for index, row in df.iterrows():
        logger.info(f"Processing video for {row['title']}")
        create_video(row)

if __name__ == '__main__':
    main()