import os
import logging
import json
from sqlalchemy import create_engine
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scripts/video_script.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_script(show_title: str, facts: list):
    """Generate a video script for a given show."""
    try:
        script = f"Welcome to our video about {show_title}!\n\n"
        script += "Here are some fun facts:\n"
        for i, fact in enumerate(facts[:3], 1):
            script += f"{i}. {fact.strip()}\n"
        script += "\nThanks for watching! Subscribe for more!"
        return script
    except Exception as e:
        logger.error(f"Error generating script for {show_title}: {e}")
        return f"Welcome to our video about {show_title}!\n\nNo facts available.\n\nThanks for watching!"

def main():
    try:
        # Load facts
        facts_path = os.path.join('generative', 'generated_facts.txt')
        if os.path.exists(facts_path):
            with open(facts_path, 'r', encoding='utf-8') as f:
                facts = f.readlines()
        else:
            logger.warning("Facts file not found, using default")
            facts = ["This show is a classic!"]
        
        # Load shows from database
        engine = create_engine('sqlite:///db/local.db')
        df = pd.read_sql('SELECT title, actors FROM shows', engine)
        logger.info(f"Found {len(df)} shows for script generation")
        if df.empty:
            logger.error("No shows found in database")
            return
        
        os.makedirs('scripts', exist_ok=True)
        for _, row in df.iterrows():
            show_title = row['title']
            script = generate_script(show_title, facts)
            output_file = os.path.join('scripts', f"{show_title.replace(' ', '_')}_script.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(script)
            logger.info(f"Generated script for {show_title} at {output_file}")
    except Exception as e:
        logger.error(f"Script generation error: {e}")

if __name__ == '__main__':
    main()