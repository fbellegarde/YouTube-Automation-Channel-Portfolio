import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml/train_recommender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_db_field(data: str) -> list:
    """Safely parses a JSON list string from the database."""
    if pd.notnull(data) and data != 'N/A' and data.strip():
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return []
    return []

def train_recommender():
    try:
        os.makedirs('ml', exist_ok=True)
        engine = create_engine('sqlite:///db/local.db')
        
        df = pd.read_sql_query('SELECT title, genres, summary FROM shows', engine)
        
        logger.info(f"Found {len(df)} shows in database for recommender training")
        if df.empty:
            logger.error("No data found in 'shows' table")
            return
            
        # Clean and combine genres and summary for TF-IDF
        def combine_text(row):
            text_parts = []
            
            # Process Genres (stored as JSON string)
            genres_list = parse_db_field(row['genres'])
            text_parts.extend(genres_list)
                
            # Process Summary
            summary = row['summary']
            if pd.notnull(summary) and summary != 'N/A' and summary.strip():
                text_parts.append(summary)
            
            return ' '.join(text_parts).lower()

        df['text'] = df.apply(combine_text, axis=1)
        
        # Filter out rows with empty text
        df = df[df['text'].str.strip().astype(bool)]
        
        if df.empty:
            logger.error("No valid data remaining after text combination.")
            return

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['text'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        model = {
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'cosine_sim': cosine_sim,
            'titles': df['title'].tolist()
        }
        
        model_path = 'ml/recommender_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Trained recommender model for {len(df)} titles.")
        logger.info(f"Saved recommender model to {model_path}")
        
    except Exception as e:
        logger.error(f"Recommender training error: {e}")

def main():
    os.makedirs('ml', exist_ok=True)
    train_recommender()

if __name__ == '__main__':
    main()