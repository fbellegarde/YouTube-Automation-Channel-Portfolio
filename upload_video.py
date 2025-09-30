import os
import logging
import json
import time
from typing import Dict, Any
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from etl.youtube_auth import get_youtube_service
from sqlalchemy import create_engine
import re
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def normalize_title(title: str) -> str:
    """Normalizes titles for matching by replacing underscores with spaces."""
    return title.replace('_', ' ').replace('.mp4', '')

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(HttpError),
    before_sleep=lambda retry_state: logger.info(f"Retrying upload for {retry_state.args[2]} due to {retry_state.outcome.exception()}")
)
def upload_video(youtube, show_data: Dict[str, Any], video_file: str):
    """Upload a video to YouTube with retry logic."""
    if not youtube or not show_data or not os.path.exists(video_file):
        logger.error(f"Invalid upload input for {video_file}")
        return False
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
        return True
    except HttpError as e:
        if 'uploadLimitExceeded' in str(e):
            logger.error(f"Upload limit exceeded for {video_file}. Adding to queue.")
            queue_upload(video_file, show_data)
            return False
        logger.error(f"Upload error for {video_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading {video_file}: {e}")
        return False

def queue_upload(video_file: str, show_data: Dict[str, Any]):
    """Save failed upload to queue for later retry."""
    queue_file = 'upload_queue.json'
    queue = []
    if os.path.exists(queue_file):
        with open(queue_file, 'r', encoding='utf-8') as f:
            queue = json.load(f)
    queue.append({'video_file': video_file, 'show_data': show_data})
    with open(queue_file, 'w', encoding='utf-8') as f:
        json.dump(queue, f, indent=2)
    logger.info(f"Added {video_file} to upload queue: {queue_file}")

def process_upload_queue(youtube):
    """Process queued uploads from upload_queue.json."""
    queue_file = 'upload_queue.json'
    if not os.path.exists(queue_file):
        logger.info("No upload queue found.")
        return
    with open(queue_file, 'r', encoding='utf-8') as f:
        queue = json.load(f)
    if not queue:
        logger.info("Upload queue is empty.")
        return
    remaining_queue = []
    for item in queue:
        video_file = item['video_file']
        show_data = item['show_data']
        logger.info(f"Retrying upload for {video_file}")
        if upload_video(youtube, show_data, video_file):
            logger.info(f"Successfully uploaded queued video: {video_file}")
        else:
            remaining_queue.append(item)
    with open(queue_file, 'w', encoding='utf-8') as f:
        json.dump(remaining_queue, f, indent=2)
    logger.info(f"Updated upload queue: {len(remaining_queue)} videos remaining")

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
    
    # Process queued uploads first
    process_upload_queue(youtube)
    
    # Process new videos
    for video_file in os.listdir(videos_dir):
        if video_file.endswith('.mp4'):
            show_title = normalize_title(video_file)
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
            time.sleep(2)  # Avoid API rate limits

if __name__ == '__main__':
    main()