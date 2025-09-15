import pandas as pd

# --- Load ETF dataset ---
dataset_file = "etf_sector_dataset_2000_2021.csv"
etf_df = pd.read_csv(dataset_file, parse_dates=["Date"])
etf_df.set_index("Date", inplace=True)

# --- Define crash start dates ---
major_crashes = pd.to_datetime(["2000-03-01", "2007-10-01", "2020-02-20"])
minor_crashes = pd.to_datetime(["2001-09-01", "2011-04-01", "2015-08-01", "2018-02-01", "2021-05-01"])

# --- Initialize Bubble_Risk column ---
etf_df["Bubble_Risk"] = "Low"

# --- Assign risk for major crashes ---
for crash in major_crashes:
    etf_df.loc[(etf_df.index >= crash - pd.DateOffset(years=1)) & (etf_df.index < crash - pd.DateOffset(months=3)), "Bubble_Risk"] = "Medium"
    etf_df.loc[(etf_df.index >= crash - pd.DateOffset(months=3)) & (etf_df.index < crash), "Bubble_Risk"] = "High"

# --- Assign risk for minor crashes ---
for crash in minor_crashes:
    etf_df.loc[(etf_df.index >= crash - pd.DateOffset(months=3)) & (etf_df.index < crash - pd.DateOffset(months=1)), "Bubble_Risk"] = "Medium"
    etf_df.loc[(etf_df.index >= crash - pd.DateOffset(months=1)) & (etf_df.index < crash), "Bubble_Risk"] = "High"

# --- Load articles dataset ---
articles_file = "fnspid_grouped_translated.csv"
articles_df = pd.read_csv(articles_file, parse_dates=["Date"])

# Concatenate multiple articles per day into a single string
articles_grouped = articles_df.groupby("Date")["Article_title"].apply(lambda x: " ".join(x)).to_frame()
articles_grouped.index.name = "Date"

# --- Merge ETF data with articles ---
merged_df = etf_df.join(articles_grouped, how="left")

# Fill missing article entries with empty string
merged_df["Article_title"] = merged_df["Article_title"].fillna("")

# --- Reset index so Date becomes a column ---
merged_df = merged_df.reset_index()

# --- Add dummy series identifier ---
merged_df["Series_ID"] = "all_etfs"

# --- Save final cleaned CSV with quoting and UTF-8 encoding ---
output_file = "dataset_2000.csv"
merged_df.to_csv(output_file, index=False, quoting=1, encoding="utf-8")
print(f"Merged dataset saved as '{output_file}'")
