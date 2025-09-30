import logging
import markovify
from sqlalchemy import create_engine
import pandas as pd
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generative/generative_facts.log'),
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
            logger.warning(f"Failed to decode JSON data: {data[:50]}...")
    return []

def generate_facts():
    os.makedirs('generative', exist_ok=True)
    try:
        engine = create_engine('sqlite:///db/local.db')
        df = pd.read_sql_query('SELECT title, summary, actors FROM shows', engine)
        
        logger.info(f"Found {len(df)} shows in database for fact generation")
        if df.empty:
            logger.error("No data found in 'shows' table")
            return ["This show is a classic!"]
        
        text_corpus = []
        for _, row in df.iterrows():
            summary = row['summary'] if pd.notnull(row['summary']) and row['summary'] != 'N/A' and row['summary'].strip() else f"{row['title']} is a beloved show."
            
            # Safely parse the JSON string for actors
            cast_list = parse_db_field(row['actors'])
            cast_text = ', '.join(cast_list) if cast_list else 'talented actors'
            
            text_corpus.append(f"{summary} Featuring {cast_text}.")
        
        if not text_corpus:
            logger.warning("No valid data for Markov model corpus")
            return ["This show is a classic!"]
        
        # Build Markov model
        text_model = markovify.Text(" ".join(text_corpus), state_size=2, well_formed=False)
        logger.info("Built Markov model")
        
        # Generate facts
        facts = []
        # Generate until 5 facts are collected or 100 tries are reached
        tries = 0
        while len(facts) < 5 and tries < 100:
            # max_chars=140 is a good length for social media snippets
            fact = text_model.make_short_sentence(max_chars=140, tries=100) 
            if fact:
                facts.append(fact)
            tries += 1

        if not facts:
            logger.warning("No facts generated after 100 tries")
            facts = ["This show is a classic!"]
            
        logger.info(f"Generated {len(facts)} fun facts.")
        return facts
    except Exception as e:
        logger.error(f"Fact generation error: {e}")
        return ["This show is a classic!"]

def main():
    facts = generate_facts()
    os.makedirs('generative', exist_ok=True)
    output_path = 'generative/generated_facts.txt'
    with open(output_path, 'w') as f:
        f.write("\n".join(facts))
    logger.info(f"Saved facts to {output_path}")

if __name__ == '__main__':
    main()