import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "social_media_usage.csv")

df = pd.read_csv(CSV_PATH)

def usage_summary():
    return {
        "total_users": int(len(df)),
        "average_posts_per_day": round(df["Posts_Per_Day"].mean(), 2),
        "average_likes_per_day": round(df["Likes_Per_Day"].mean(), 2),
        "average_follows_per_day": round(df["Follows_Per_Day"].mean(), 2),
    }
