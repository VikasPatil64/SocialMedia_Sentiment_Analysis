import pandas as pd

# Load dataset (update file name if different)
file_path = "data/sentiment140.csv"

# Define column names since dataset has no header
columns = ["target", "id", "date", "flag", "user", "text"]
df = pd.read_csv(file_path, encoding="latin-1", names=columns)

# Keep only target and text
df = df[["target", "text"]]

# Map sentiment values
sentiment_map = {0: "negative", 2: "neutral", 4: "positive"}
df["sentiment"] = df["target"].map(sentiment_map)
df = df.drop(columns=["target"])

# Drop null or empty rows
df = df.dropna(subset=["text"])

# Save cleaned data
df.to_csv("data/cleaned_sentiment140.csv", index=False)
print("✅ Cleaned Sentiment140 dataset saved to data/cleaned_sentiment140.csv")
print(df.head())
