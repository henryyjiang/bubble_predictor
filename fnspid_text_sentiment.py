import pandas as pd
from textblob import TextBlob  # simple sentiment analysis

# --- Load merged dataset ---
df = pd.read_csv("market_news_2025.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# --- Process Article_title column ---
# 1. Fill missing articles with empty string
df["Article_title"] = df["Article_title"].fillna("")

# 2. Calculate daily sentiment (average polarity of all text)
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 (negative) to +1 (positive)

df["Article_Sentiment"] = df["Article_title"].apply(get_sentiment)

# 3. Count number of articles per day
df["Article_Count"] = df["Article_title"].apply(lambda x: len(x.split('.')))

# 4. Drop raw text column (Vertex cannot use text covariate)
df = df.drop(columns=["Article_title"])

# --- Optional: add series identifier for Vertex ---
df["Series_ID"] = "all_etfs"

# --- Save cleaned dataset ---
df.to_csv("market_news_2025.csv", index=True)
print("Saved numeric-only dataset for Vertex AI as 'dataset_2000.csv'")
