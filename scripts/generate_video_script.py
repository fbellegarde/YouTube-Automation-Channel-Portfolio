import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('scripts', 'video_script.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_script(show_title: str):
    try:
        with open(os.path.join('generative', 'generated_facts.txt'), 'r') as f:
            facts = f.readlines()
    except Exception as e:
        logger.error(f"Error reading facts: {e}")
        facts = ["Sample fact about the show."]

    script = f"Welcome to our video about {show_title}!\n\n"
    script += "Here are some fun facts:\n"
    for i, fact in enumerate(facts[:3], 1):
        script += f"{i}. {fact.strip()}\n"
    script += "\nThanks for watching! Subscribe for more!"

    try:
        os.makedirs('scripts', exist_ok=True)
        output_file = os.path.join('scripts', 'video_script.txt')
        with open(output_file, 'w') as f:
            f.write(script)
        logger.info(f"Generated script for {show_title} at {output_file}")
    except Exception as e:
        logger.error(f"Error saving script: {e}")
        raise

if __name__ == '__main__':
    generate_script("SpongeBob SquarePants")  # Example