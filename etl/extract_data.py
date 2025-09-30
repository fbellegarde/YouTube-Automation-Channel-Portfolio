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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extract_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Base = declarative_base()

class Show(Base):
    __tablename__ = 'shows'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    genres = Column(Text)  # JSON string
    summary = Column(Text)
    cast = Column(Text)  # JSON string
    year = Column(String)
    imdb_rating = Column(String)
    network = Column(String)
    premiered = Column(String)
    status = Column(String)
    image_url = Column(String)

def fetch_omdb_data(title: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not title or not api_key:
        logger.error("Invalid OMDB input")
        return None
    url = f'http://www.omdbapi.com/?t={requests.utils.quote(title)}&type=series&apikey={api_key}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('Response') == 'True':
            return {
                'title': data.get('Title', ''),
                'genres': data.get('Genre', ''),
                'summary': data.get('Plot', ''),
                'cast': data.get('Actors', ''),
                'year': data.get('Year', ''),
                'imdb_rating': data.get('imdbRating', 'N/A'),
                'Poster': data.get('Poster', 'N/A')
            }
        else:
            logger.error(f"OMDB error: {data.get('Error')}")
            return None
    except Exception as e:
        logger.error(f"OMDB fetch error: {e}")
        return None

def fetch_tvmaze_data(title: str) -> Optional[Dict[str, Any]]:
    if not title:
        logger.error("Invalid TVMaze input")
        return None
    url = f'http://api.tvmaze.com/singlesearch/shows?q={requests.utils.quote(title)}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            'network': data.get('network', {}).get('name', 'N/A'),
            'premiered': data.get('premiered', 'N/A'),
            'status': data.get('status', 'N/A')
        }
    except Exception as e:
        logger.error(f"TVMaze fetch error: {e}")
        return None

def scrape_wikipedia_image(title: str, omdb_data: Dict[str, Any]) -> Optional[str]:
    if not title:
        logger.error("Invalid Wikipedia input")
        return None
    poster_url = omdb_data.get('Poster', 'N/A')
    if poster_url != 'N/A':
        logger.info(f"Using OMDB Poster for {title}: {poster_url}")
        return poster_url
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
                img_url = f"https:{img_tag['src']}"
                img_url = re.sub(r'/thumb/', '/', img_url)
                img_url = re.sub(r'/\d+px-[^/]+$', '', img_url)
                logger.info(f"Using Wikipedia image for {title}: {img_url}")
                return img_url
        logger.warning(f"No image found for {title} on Wikipedia")
        return None
    except Exception as e:
        logger.error(f"Wikipedia scrape error: {e}")
        return None

def extract_show_data(title: str) -> Dict[str, Any]:
    data = {'title': title, 'genres': '', 'summary': '', 'cast': '', 'year': '', 'imdb_rating': 'N/A', 'network': 'N/A', 'premiered': 'N/A', 'status': 'N/A', 'image_url': 'N/A'}
    api_key = os.getenv('OMDB_API_KEY')
    omdb_data = fetch_omdb_data(title, api_key)
    if omdb_data:
        data.update(omdb_data)
    tvmaze_data = fetch_tvmaze_data(title)
    if tvmaze_data:
        data.update(tvmaze_data)
    data['image_url'] = scrape_wikipedia_image(title, omdb_data) or 'N/A'
    return data

def load_to_db(data: Dict[str, Any]):
    engine = create_engine('sqlite:///db/local.db')
    Base.metadata.drop_all(engine, [Base.metadata.tables['shows']])
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        show = Show(
            title=data['title'],
            genres=json.dumps(data['genres'].split(', ') if data['genres'] else []),
            summary=data['summary'],
            cast=json.dumps(data['cast'].split(', ') if data['cast'] else []),
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
        logger.error(f"DB load error: {e}")

def main():
    shows = ['The Powerpuff Girls', 'Hannah Montana', 'SpongeBob SquarePants', 'Rugrats', 'Hey Arnold!', 'Dexter\'s Laboratory', 'The Fairly OddParents']
    os.makedirs('output', exist_ok=True)
    for show in shows:
        data = extract_show_data(show)
        if not data.get('title'):
            continue
        json_file = f'output/{show.replace(" ", "_")}.json'
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        load_to_db(data)
        time.sleep(2)

if __name__ == '__main__':
    main()