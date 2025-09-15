import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from textblob import TextBlob

TIINGO_API_KEY = "a05c584aa9e153e081f698bb9abcabaeae877fd4"
days = 30
use_tiingo = True

etfs = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
    "GLD": "Gold",
    "^VIX": "VIX",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLV": "Health Care"
}


# --- Option 1: Specific time range before today ---
days = 30  # 1 year back
end_date = datetime.today()
start_date = end_date - timedelta(days=days)

print("Relative range:", start_date.date(), "to", end_date.date())

# --- Option 2: Two specific days ---
#start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
#end_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

print("Fixed range:", start_date.date(), "to", end_date.date())
all_data = []

for ticker, name in etfs.items():
    print(f"Fetching {name} ({ticker}) from {start_date.date()} to {end_date.date()}...")

    df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)

    if df.empty:
        print(f"Warning: No data returned for {ticker}")
        continue

    df = df[["Close"]].copy()
    df.rename(columns={"Close": f"{ticker}_AdjClose"}, inplace=True)

    # Daily return (%) for momentum
    df[f"{ticker}_Return"] = df[f"{ticker}_AdjClose"].pct_change() * 100

    # Rolling 20-day volatility (%)
    df[f"{ticker}_Vol_20d"] = df[f"{ticker}_Return"].rolling(20).std()

    all_data.append(df)

# Merge all ETFs on date
if all_data:
    merged_data = pd.concat(all_data, axis=1)

    # Flatten columns to avoid MultiIndex issues
    merged_data.columns = [col if isinstance(col, str) else col[0] for col in merged_data.columns]

    # Add relative price ratios vs SPY
    for ticker in ["QQQ", "DIA", "XLK", "XLF", "XLE", "XLI", "XLY", "XLV", "GLD"]:
        if f"{ticker}_AdjClose" in merged_data.columns and "SPY_AdjClose" in merged_data.columns:
            merged_data[f"{ticker}_to_SPY"] = merged_data[f"{ticker}_AdjClose"] / merged_data["SPY_AdjClose"]

    # Add SPY and QQQ 1-week forward returns (%) as prediction targets
    forward_days = 5  # ~1 trading week
    for ticker in ["SPY", "QQQ", "DIA", "GLD", "^VIX"]:
        col_name = ticker.replace("^", "") + "_1wChange"
        if f"{ticker}_AdjClose" in merged_data.columns:
            merged_data[col_name] = (merged_data[f"{ticker}_AdjClose"].shift(-forward_days) - merged_data[
                f"{ticker}_AdjClose"]) / merged_data[f"{ticker}_AdjClose"] * 100
else:
    merged_data = pd.DataFrame()
    print("No data was fetched.")

for col in merged_data.columns:
    if col.startswith('^VIX'):
        new_col = col.replace('^VIX', 'VIX')
        merged_data.rename(columns={col: new_col}, inplace=True)

print("DataFrame prepared for last 30 days:")
print(merged_data.head())


# Check if Tiingo API key is valid
if not TIINGO_API_KEY or TIINGO_API_KEY == "YOUR API KEY HERE":
    print("No valid Tiingo API key provided. Article sentiment will default to 0.")
    use_tiingo = False

if use_tiingo:
    headers = {"Content-Type": "application/json", "Authorization": f"Token {TIINGO_API_KEY}"}
    all_articles = []

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"https://api.tiingo.com/tiingo/news?startDate={date_str}&endDate={date_str}&limit=50"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            news_batch = response.json()
        except Exception as e:
            print(f"Error fetching news for {date_str}: {e}")
            news_batch = []

        for item in news_batch:
            all_articles.append({
                "Date": date_str,
                "Headline": item.get("title", "")
            })

        print(f"Fetched {len(news_batch)} articles for {date_str}")
        current_date += timedelta(days=1)

    df_news = pd.DataFrame(all_articles)
    df_grouped = df_news.groupby("Date")["Headline"].apply(lambda x: " ".join([str(h) for h in x])).reset_index()

    def get_sentiment(text):
        try:
            return TextBlob(text).sentiment.polarity
        except:
            return 0.0

    df_grouped["Article_Sentiment"] = df_grouped["Headline"].apply(get_sentiment)
else:
    # No API key: create empty sentiment dataframe
    df_grouped = pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date),
        "Article_Sentiment": 0.0
    })
    print("df grouped", df_grouped)

if use_tiingo:
    df_news = pd.DataFrame(all_articles)
    df_grouped = df_news.groupby("Date")["Headline"].apply(lambda x: " ".join([str(h) for h in x])).reset_index()

    # Compute sentiment
    df_grouped["Article_Sentiment"] = df_grouped["Headline"].apply(lambda t: TextBlob(t).sentiment.polarity)
else:
    # No API key: create sentiment df with zeros
    df_grouped = pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date),
        "Article_Sentiment": 0.0
    })

if not use_tiingo:
    df_grouped['Date'] = df_grouped['Date'].dt.date

df_grouped["Series_ID"] = "all_etfs"
df_grouped = df_grouped.drop(columns=["Headline"], errors='ignore')
df_grouped['Date'] = pd.to_datetime(df_grouped['Date'])

print(df_grouped.head())


merged_data = merged_data.reset_index()  # moves index to column
merged_data.rename(columns={'index': 'Date'}, inplace=True)

df_grouped['Date'] = pd.to_datetime(df_grouped['Date'])
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

final_df = pd.merge(merged_data, df_grouped, on='Date', how='left')
final_df.set_index('Date', inplace=True)

final_df.to_csv("dataset_past_month.csv", index=True, encoding="utf-8-sig")
print(f"Saved merged dataset: {final_df.shape[0]} rows, {final_df.shape[1]} columns")