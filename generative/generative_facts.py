import os
import markovify
from sqlalchemy import create_engine
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('generative', 'generative_facts.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load data
try:
    engine = create_engine('sqlite:///db/local.db')
    df = pd.read_sql('shows', engine)
    if df.empty:
        logger.error("No data found in shows table")
        raise ValueError("Empty shows table")
except Exception as e:
    logger.error(f"Database error: {e}")
    raise

# Generate Markov model
try:
    text = ' '.join(df['summary'])
    if not text.strip():
        logger.error("No valid summary text for Markov model")
        raise ValueError("Empty summary text")
    model = markovify.Text(text, state_size=2)
    logger.info("Built Markov model")
except Exception as e:
    logger.error(f"Markov model error: {e}")
    raise

# Generate facts
generated = []
try:
    for _ in range(5):
        fact = model.make_sentence(tries=200)  # Was tries=100
        if fact:
            generated.append(fact)
    if not generated:
        logger.warning("No facts generated")
        generated = ["Sample fact: This show is a classic!"]
    logger.info("Generated fun facts:")
    for fact in generated:
        logger.info(fact)
except Exception as e:
    logger.error(f"Fact generation error: {e}")
    raise

# Save facts
try:
    os.makedirs('generative', exist_ok=True)
    output_file = os.path.join('generative', 'generated_facts.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(generated))
    logger.info(f"Saved facts to {output_file}")
except Exception as e:
    logger.error(f"Save error: {e}")
    raise