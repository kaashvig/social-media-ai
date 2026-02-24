import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "social_media_usage.csv")

print("CSV PATH:", CSV_PATH)   # debug (keep temporarily)

df = pd.read_csv(CSV_PATH)
def usage_summary():
    import os
    import pandas as pd

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    CSV_PATH = os.path.join(PROJECT_ROOT, "data", "social_media_usage.csv")

    df = pd.read_csv(CSV_PATH)

    return {
        "total_users": len(df),
        "average_posts_per_day": df["Posts_Per_Day"].mean(),
        "average_likes_per_day": df["Likes_Per_Day"].mean(),
        "average_follows_per_day": df["Follows_Per_Day"].mean(),
    }