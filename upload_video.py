import os
import logging
from typing import Dict, Any
from googleapiclient.http import MediaFileUpload
from etl.youtube_auth import get_youtube_service
from sqlalchemy import create_engine
import re
import json

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
    if not youtube or not show_data or not os.path.exists(video_file):
        logger.error("Invalid upload input")
        return
    try:
        summary = re.sub(r'<[^>]+>', '', show_data.get('summary', ''))
        title = show_data.get('title', 'Unknown')[:90] + ' Highlights'
        description = f"Featuring: {show_data.get('cast', '')}\nSummary: {summary}"[:5000]
        tags = show_data.get('genres', '').split(', ') if show_data.get('genres') else []
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags + ['90s cartoons', 'nostalgia', 'kids shows'],
                'categoryId': '24'
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
    youtube = get_youtube_service()
    videos_dir = 'videos'
    engine = create_engine('sqlite:///db/local.db')
    os.makedirs(videos_dir, exist_ok=True)
    for video_file in os.listdir(videos_dir):
        if video_file.endswith('.mp4'):
            show_title = video_file.replace('.mp4', '').replace('_', ' ')
            # Load show data from JSON
            json_file = f'output/{video_file.replace(".mp4", "")}.json'
            show_data = {
                'title': show_title,
                'summary': f"Explore the iconic show {show_title}!",
                'cast': 'N/A',
                'genres': 'Animation, Comedy',
                'network': 'N/A',
                'premiered': 'N/A',
                'status': 'N/A'
            }
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    show_data.update(json.load(f))
            video_path = os.path.join(videos_dir, video_file)
            upload_video(youtube, show_data, video_path)

if __name__ == '__main__':
    main()