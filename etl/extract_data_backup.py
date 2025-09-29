import wikipediaapi  # For safe Wikipedia access
import pandas as pd
import json
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Why: WikipediaAPI is a safe, official wrapper using HTTPS. No viruses, respects rate limits.

# List of shows (expand as needed)
shows = ["The Powerpuff Girls", "Hannah Montana", "SpongeBob SquarePants", "Kim Possible", "Rugrats"]

def extract_show_data(show_name):
    wiki = wikipediaapi.Wikipedia(user_agent='YouTubeAutomationETL/1.0 (fernandobellegarde64@gmail.com)', language='en')  # English Wikipedia with proper User-Agent
    page = wiki.page(show_name)
    if not page.exists():
        print(f"Warning: Page for '{show_name}' does not exist.")
        return None
    # Extract sections: Summary, directors, actors, fun facts (from infobox or sections)
    data = {
        'title': page.title,
        'summary': page.summary,
        'sections': {section.title: section.text for section in page.sections if 'Cast' in section.title or 'Production' in section.title or 'Trivia' in section.title}
    }
    return data

# ETL Step 1: Extract
raw_data = []
for show in shows:
    data = extract_show_data(show)
    if data:
        raw_data.append(data)

if not raw_data:
    print("No data extracted—check show names or network.")
    # Exit or handle gracefully in production

# Save raw to JSON (for S3 later)
with open('data/raw_shows.json', 'w') as f:
    json.dump(raw_data, f)

# ETL Step 2: Transform (clean to DataFrame)
df = pd.DataFrame(raw_data)
df['fun_facts'] = df['sections'].apply(lambda x: x.get('Trivia', '') if x else '')  # Example transform
df = df.drop('sections', axis=1)  # Clean

# ETL Step 3: Load to SQLite
Base = declarative_base()

class Show(Base):
    __tablename__ = 'shows'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    summary = Column(Text)
    fun_facts = Column(Text)

engine = create_engine('sqlite:///db/local.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

for _, row in df.iterrows():
    show = Show(title=row['title'], summary=row['summary'], fun_facts=row['fun_facts'])
    session.add(show)

try:
    session.commit()
    print("ETL Complete: Data loaded to local.db")
except Exception as e:
    session.rollback()
    print(f"Database error: {e}")