import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import pandas as pd

# 1. Load API key from .env file
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# 2. Create a YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

# 3. Function to fetch comments from a video
def get_video_comments(video_id, max_results=20):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

# 4. Test run
if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    parsed_url = urlparse(video_url)

    if "youtu.be" in parsed_url.netloc:
        video_id = parsed_url.path[1:]  # everything after '/'
    else:
        video_id = parse_qs(parsed_url.query).get("v", [None])[0] # extract ID from URL
    comments = get_video_comments(video_id)
    print("\n--- Comments Fetched ---")
    for c in comments:
        print("-", c)
# Create folder if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")
# After fetching comments
df = pd.DataFrame(comments, columns=["comment"])
df.to_csv("data/yt_comments.csv", index=False)
print(f"\nSaved {len(df)} comments to data/yt_comments.csv")