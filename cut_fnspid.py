import pandas as pd
import csv
from langdetect import detect

# Configuration
input_file = "fnspid.csv"
output_file = "fnspid_cleaned_grouped.csv"
chunksize = 50000  # number of rows per chunk (adjust based on RAM)
sample_frac = 0.1  # fraction of rows to keep
keep_english_only = False  # set False if you want all languages


# Function to detect English
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False


# List to store sampled chunks
sampled_chunks = []

# Columns to keep
cols_to_keep = ["Date", "Article_title"]

# Read CSV in chunks
for chunk in pd.read_csv(input_file, usecols=cols_to_keep,
                         chunksize=chunksize, encoding="utf-8", quoting=csv.QUOTE_ALL):

    # Randomly sample rows
    chunk_sampled = chunk.sample(frac=sample_frac, random_state=42)

    # Optionally filter English only
    if keep_english_only:
        chunk_sampled = chunk_sampled[chunk_sampled["Article_title"].apply(is_english)]

    # Keep chunk
    sampled_chunks.append(chunk_sampled)

# Concatenate all chunks
df_reduced = pd.concat(sampled_chunks).reset_index(drop=True)

# Convert Date to datetime and strip time
df_reduced["Date"] = pd.to_datetime(df_reduced["Date"]).dt.date

# Group by Date and combine article titles into a single string per day
df_grouped = df_reduced.groupby("Date")["Article_title"].apply(
    lambda x: " ".join(x.dropna().astype(str))
).reset_index()
# Save cleaned and grouped dataset
df_grouped.to_csv(output_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

print(f"Final dataset shape: {df_grouped.shape}")
print(f"Saved cleaned and grouped dataset as: {output_file}")
