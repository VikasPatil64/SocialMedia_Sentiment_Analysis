import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

# Load cleaned comments
df = pd.read_csv("data/cleaned_yt_comments.csv")

# Drop empty or NaN comments before sentiment analysis
df = df.dropna(subset=["cleaned_comment"])

sia = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_sentiment(text):
    if pd.isna(text):  # handle NaN
        text = ""
    text = str(text)  # make sure it's a string
    score = sia.polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


# Apply sentiment analysis
df["sentiment"] = df["cleaned_comment"].apply(get_sentiment)

# Print each comment with color-coded sentiment
from colorama import Fore, Style

for i, row in df.iterrows():
    sentiment = row["sentiment"]
    if sentiment == "Positive":
        color = Fore.GREEN
    elif sentiment == "Negative":
        color = Fore.RED
    else:
        color = Fore.YELLOW  # Neutral

    print(f"{color}Comment: {row['cleaned_comment']}")
    print(f"Sentiment: {sentiment}{Style.RESET_ALL}")
    print("-" * 50)
# Save results
df.to_csv("data/sentiments_yt_comments.csv", index=False)
print("Sentiment analysis done! Results saved in sentiments_yt_comments.csv")

# --- Count summary ---
summary = (
    df.groupby("sentiment")
    .size()
    .reset_index(name="count")
)

print("\nSentiment Count Summary:")
print(summary.to_string(index=False))

# Create summary counts of sentiments per video (if video_id available)
summary = df.groupby("sentiment").size().reset_index(name="count")

# Save summary to CSV
summary_file = "data/sentiment_summary.csv"
summary.to_csv(summary_file, index=False)

print(f"\nSentiment summary saved to {summary_file}")
print(summary)

