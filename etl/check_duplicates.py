import pandas as pd

df = pd.read_csv('data/shows_data.csv')
print("Total rows:", len(df))
print("Unique titles:", df['title'].nunique())
print("Duplicates:", df[df.duplicated(subset=['title'], keep=False)])