import pandas as pd
import sqlite3
import json

conn = sqlite3.connect('db/local.db')
df = pd.read_sql_query("SELECT * FROM shows", conn)
conn.close()

# Parse JSON fields
df['genres'] = df['genres'].apply(json.loads)
df['cast'] = df['cast'].apply(json.loads)

print("Data Quality Report:")
print(f"Total shows: {len(df)}")
print(f"Unique titles: {df['title'].nunique()}")
print(f"Empty fields: {df[['genres', 'cast', 'premiered', 'network', 'image_url']].isna().sum().to_dict()}")
print(f"Avg summary length: {df['summary'].str.len().mean():.1f} chars")
print(df[['title', 'cast', 'premiered', 'genres', 'image_url']].head())