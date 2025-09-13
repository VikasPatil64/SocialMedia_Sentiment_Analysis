import pandas as pd
import re
from nltk.corpus import stopwords

# Load saved comments
df = pd.read_csv("data/yt_comments.csv")

# Download stopwords first (if not done yet)
import nltk
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_comment(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special chars
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

df["cleaned_comment"] = df["comment"].apply(clean_comment)

# Save cleaned comments
df.to_csv("data/cleaned_yt_comments.csv", index=False)
print("Cleaned comments saved!")
