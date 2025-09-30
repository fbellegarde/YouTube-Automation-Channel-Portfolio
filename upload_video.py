import os
import logging
from typing import Dict, Any
from googleapiclient.http import MediaFileUpload
from etl.youtube_auth import get_youtube_service
from sqlalchemy import create_engine
import re
import json
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def upload_video(youtube, show_data: Dict[str, Any], video_file: str):
    """Upload a video to YouTube."""
    if not youtube or not show_data or not os.path.exists(video_file):
        logger.error("Invalid upload input")
        return
    try:
        summary = re.sub(r'<[^>]+>', '', show_data.get('summary', ''))
        title = show_data.get('title', 'Unknown')[:90] + ' Highlights'
        try:
            actors_list = json.loads(show_data.get('actors', '[]')) if show_data.get('actors') != 'N/A' else ['N/A']
            actors_text = ', '.join(actors_list)
        except json.JSONDecodeError:
            actors_text = 'N/A'
        description = f"Featuring: {actors_text}\nSummary: {summary}"[:5000]
        tags = json.loads(show_data.get('genres', '[]')) if show_data.get('genres') != 'N/A' else []
        tags.extend(['90s cartoons', 'nostalgia', 'kids shows'])
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': '24'  # Entertainment
            },
            'status': {'privacyStatus': 'public'}
        }
        request = youtube.videos().insert(
            part='snippet,status',
            body=body,
            media_body=MediaFileUpload(video_file, resumable=True)
        )
        response = request.execute()
        logger.info(f"Uploaded {title}: https://youtu.be/{response['id']}")
    except Exception as e:
        logger.error(f"Upload error: {e}")

def main():
    """Main function to upload videos."""
    youtube = get_youtube_service()
    if not youtube:
        logger.error("Failed to initialize YouTube service")
        return
    videos_dir = 'videos'
    engine = create_engine('sqlite:///db/local.db')
    os.makedirs(videos_dir, exist_ok=True)
    try:
        df = pd.read_sql('SELECT title, summary, actors, genres FROM shows', engine)
        logger.info(f"Found {len(df)} shows in database")
        if df.empty:
            logger.error("No shows found in database")
            return
    except Exception as e:
        logger.error(f"Database query error: {e}")
        return
    for video_file in os.listdir(videos_dir):
        if video_file.endswith('.mp4'):
            show_title = video_file.replace('.mp4', '').replace('_', ' ')
            show_data = df[df['title'] == show_title].to_dict('records')
            if not show_data:
                logger.warning(f"No database entry for {show_title}, using default")
                show_data = [{
                    'title': show_title,
                    'summary': f"Explore the iconic show {show_title}!",
                    'actors': '[]',
                    'genres': '["Animation", "Comedy"]'
                }]
            video_path = os.path.join(videos_dir, video_file)
            upload_video(youtube, show_data[0], video_path)

if __name__ == '__main__':
    main()