import os
import requests
import json
import logging
import time
from typing import Dict, Any, Optional
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables (e.g., OMDB_API_KEY)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl/extract_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Base = declarative_base()

class Show(Base):
    __tablename__ = 'shows'
    id = Column(Integer, primary_key=True)
    title = Column(String, unique=True)
    # FIX: Replaced U+00A0 with standard space
    genres = Column(Text)  # JSON string 
    summary = Column(Text)
    # FIX: Replaced U+00A0 with standard space
    actors = Column(Text)  # JSON string (renamed from cast to avoid reserved keyword)
    year = Column(String)
    imdb_rating = Column(String)
    network = Column(String)
    premiered = Column(String)
    status = Column(String)
    image_url = Column(String)
    
    def __repr__(self):
        # FIX: Replaced U+00A0 with standard space
        return f"<Show(title='{self.title}')>"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_omdb_data(title: str, api_key: str) -> Optional[Dict[str, Any]]:
    # FIX: Replaced U+00A0 with standard space
    if not title or not api_key:
        logger.error("Invalid OMDB input: title or API key missing.")
        return None
    url = f'http://www.omdbapi.com/?t={requests.utils.quote(title)}&type=series&apikey={api_key}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('Response') == 'True':
            logger.info(f"Fetched OMDB data for {title}")
            return {
                'title': data.get('Title', title), # Use API title if available
                'genres': data.get('Genre', ''),
                'summary': data.get('Plot', ''),
                'actors': data.get('Actors', ''),
                'year': data.get('Year', ''),
                'imdb_rating': data.get('imdbRating', 'N/A'),
                'Poster': data.get('Poster', 'N/A') # Keep Poster URL for image check
            }
        else:
            logger.error(f"OMDB error for {title}: {data.get('Error')}")
            return None
    except Exception as e:
        logger.error(f"OMDB fetch error for {title}: {e}")
        raise # Reraise to trigger tenacity retry

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_tvmaze_data(title: str) -> Optional[Dict[str, Any]]:
    if not title:
        logger.error("Invalid TVMaze input: title missing.")
        return None
    url = f'http://api.tvmaze.com/singlesearch/shows?q={requests.utils.quote(title)}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Fetched TVMaze data for {title}")
        return {
            # FIX: Replaced U+00A0 with standard space
            'network': data.get('network', {}).get('name', 'N/A'),
            'premiered': data.get('premiered', 'N/A'),
            'status': data.get('status', 'N/A')
        }
    except Exception as e:
        logger.error(f"TVMaze fetch error for {title}: {e}")
        raise # Reraise to trigger tenacity retry

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def scrape_wikipedia_image(title: str, omdb_data: Dict[str, Any]) -> Optional[str]:
    # 1. Prioritize OMDB Poster URL
    poster_url = omdb_data.get('Poster', 'N/A')
    if poster_url and poster_url != 'N/A':
        logger.info(f"Using OMDB Poster for {title}")
        return poster_url
    
    # 2. Fallback to Wikipedia scrape
    url = f'https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(" ", "_"))}'
    headers = {'User-Agent': 'YouTubeAutomationBot/1.0 (https://example.com/contact)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        infobox = soup.find('table', class_='infobox')
        if infobox:
            img_tag = infobox.find('img')
            if img_tag and 'src' in img_tag.attrs:
                img_url = f"https:{img_tag['src']}" if img_tag['src'].startswith('//') else img_tag['src']
                # Clean up Wikipedia image URL for full-res
                img_url = re.sub(r'/thumb/', '/', img_url)
                img_url = re.sub(r'/\d+px-[^/]+$', '', img_url)
                logger.info(f"Using Wikipedia image for {title}")
                return img_url
        logger.warning(f"No image found for {title} on Wikipedia or OMDB")
        return None
    except Exception as e:
        logger.error(f"Wikipedia scrape error for {title}: {e}")
        raise # Reraise to trigger tenacity retry

def extract_show_data(title: str) -> Dict[str, Any]:
    # Initialize with defaults
    data = {'title': title, 'genres': '', 'summary': '', 'actors': '', 'year': '', 'imdb_rating': 'N/A', 'network': 'N/A', 'premiered': 'N/A', 'status': 'N/A', 'image_url': 'N/A'}
    
    api_key = os.getenv('OMDB_API_KEY')
    if not api_key:
        logger.error("OMDB_API_KEY environment variable not set. Cannot fetch data.")
        return data

    omdb_data = fetch_omdb_data(title, api_key)
    if omdb_data:
        data.update(omdb_data)
        
    # Check if a valid title was found from the API (OMDB can sometimes correct the title)
    api_title = data.get('title')
    if api_title and api_title != title:
        logger.info(f"Using API-corrected title: {api_title}")
        title = api_title
        
    # Use the potentially corrected title for TVMaze and Wikipedia
    tvmaze_data = fetch_tvmaze_data(title)
    if tvmaze_data:
        data.update(tvmaze_data)
        
    data['image_url'] = scrape_wikipedia_image(title, omdb_data if omdb_data else {}) or 'N/A'
    
    # Remove temporary 'Poster' key
    if 'Poster' in data:
        del data['Poster']
        
    return data

def load_to_db(data: Dict[str, Any]):
    engine = create_engine('sqlite:///db/local.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Check for existing entry
        existing_show = session.query(Show).filter_by(title=data['title']).first()
        if existing_show:
            logger.warning(f"Show {data['title']} already exists, skipping insert.")
            return

        # Prepare JSON fields, ensuring lists are created even from empty strings
        # Splits comma-separated string from OMDB/TVMaze into a list
        genres_list = [g.strip() for g in data['genres'].split(', ') if g.strip()]
        actors_list = [a.strip() for a in data['actors'].split(', ') if a.strip()]
        
        show = Show(
            title=data['title'],
            genres=json.dumps(genres_list), # Store as JSON string
            summary=data['summary'] if data['summary'] != 'N/A' else None,
            actors=json.dumps(actors_list), # Store as JSON string
            year=data['year'],
            imdb_rating=data['imdb_rating'],
            network=data['network'],
            premiered=data['premiered'],
            status=data['status'],
            image_url=data['image_url']
        )
        session.add(show)
        session.commit()
        logger.info(f"Loaded {data['title']} to DB")
    except Exception as e:
        session.rollback()
        logger.error(f"DB load error for {data['title']}: {e}")
    finally:
        session.close()

def main():
    os.makedirs('output', exist_ok=True)
    os.makedirs('db', exist_ok=True)
    
    engine = create_engine('sqlite:///db/local.db')
    
    # Corrected DB Teardown/Setup
    Base.metadata.drop_all(engine, tables=[Base.metadata.tables['shows']] if 'shows' in Base.metadata.tables else None)
    Base.metadata.create_all(engine)
    logger.info("Database 'shows' table refreshed.")
    
    shows = ['The Powerpuff Girls', 'Hannah Montana', 'SpongeBob SquarePants', 'Rugrats', 'Hey Arnold!', 'Dexter\'s Laboratory', 'The Fairly OddParents']
    
    for show in shows:
        logger.info(f"--- Starting extraction for: {show} ---")
        data = extract_show_data(show)
        
        # Use a safe title for the JSON file to ensure consistency
        safe_title = data['title'].replace(" ", "_").replace(":", "").replace("'", "")
        if not safe_title:
             logger.error(f"Skipping load for {show} due to failed/incomplete data extraction.")
             continue
             
        json_file = f'output/{safe_title}.json'
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        load_to_db(data)
        time.sleep(1) # Be respectful to APIs

if __name__ == '__main__':
    main()