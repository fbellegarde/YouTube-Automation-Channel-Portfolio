import logging
import markovify
from sqlalchemy import create_engine
import pandas as pd
import os

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

def generate_facts():
    try:
        engine = create_engine('sqlite:///db/local.db')
        # Try 'tv_shows' first, fall back to 'shows'
        table_name = 'tv_shows'
        try:
            df = pd.read_sql(f'SELECT title, summary, cast FROM {table_name}', engine)
        except Exception as e:
            logger.warning(f"Table 'tv_shows' not found: {e}, trying 'shows'")
            table_name = 'shows'
            df = pd.read_sql(f'SELECT title, summary, cast FROM {table_name}', engine)
        logger.info(f"Found {len(df)} shows in database")
        # Combine summary and cast for richer text
        text_corpus = []
        for _, row in df.iterrows():
            summary = row['summary'] if row['summary'] and row['summary'] != 'N/A' else f"{row['title']} is a beloved show."
            cast = row['cast'] if row['cast'] and row['cast'] != 'N/A' else 'talented actors'
            text_corpus.append(f"{summary} Featuring {cast}.")
        if not text_corpus:
            logger.warning("No valid data for Markov model")
            return ["This show is a classic!"]
        # Build Markov model
        text_model = markovify.Text(" ".join(text_corpus), state_size=2)
        logger.info("Built Markov model")
        # Generate facts
        facts = []
        for _ in range(5):
            fact = text_model.make_short_sentence(140, tries=100)
            if fact:
                facts.append(fact)
        if not facts:
            logger.warning("No facts generated")
            facts = ["This show is a classic!"]
        logger.info("Generated fun facts:")
        for fact in facts:
            logger.info(f"Sample fact: {fact}")
        return facts
    except Exception as e:
        logger.error(f"Fact generation error: {e}")
        return ["This show is a classic!"]

def main():
    facts = generate_facts()
    os.makedirs('generative', exist_ok=True)
    with open('generative/generated_facts.txt', 'w') as f:
        f.write("\n".join(facts))
    logger.info("Saved facts to generative/generated_facts.txt")

if __name__ == '__main__':
    main()