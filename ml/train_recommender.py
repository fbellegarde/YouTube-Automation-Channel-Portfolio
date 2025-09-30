import pandas as pd
import sqlite3
import json
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ml/recommender.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load data
try:
    conn = sqlite3.connect('db/local.db')
    df = pd.read_sql_query("SELECT * FROM shows", conn)
    conn.close()
except Exception as e:
    logger.error(f"Database error: {e}")
    raise

# Parse JSON fields
try:
    df['genres'] = df['genres'].apply(json.loads)
    df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))
except Exception as e:
    logger.error(f"JSON parsing error: {e}")
    raise

# Create TF-IDF matrix
try:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['summary'] + " " + df['genres_str'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logger.info("Trained recommender model")
except Exception as e:
    logger.error(f"Model training error: {e}")
    raise

# Save model
try:
    with open('ml/recommender_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, cosine_sim, df), f)
    logger.info("Saved recommender model to ml/recommender_model.pkl")
except Exception as e:
    logger.error(f"Model save error: {e}")
    raise