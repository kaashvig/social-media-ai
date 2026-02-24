import streamlit as st
import requests
import pandas as pd

API_URL = st.secrets["API_URL"]

st.set_page_config(page_title="Privacy-Preserving AI Analytics", layout="wide")

st.title("Privacy-Preserving Social Media Analytics Dashboard")
st.caption("Local AI-powered behavioral and emotional intelligence system")

st.divider()

# =========================
# SECTION 1: DATASET SUMMARY
# =========================

st.header("1. Engagement Analytics Overview")

try:
    summary = requests.get(f"{API_URL}/summary").json()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", summary["total_users"])
    col2.metric("Avg Posts/Day", summary["average_posts_per_day"])
    col3.metric("Avg Likes/Day", summary["average_likes_per_day"])
    col4.metric("Avg Follows/Day", summary["average_follows_per_day"])
except:
    st.error("Backend not reachable.")

st.divider()

# =========================
# SECTION 2: ENGAGEMENT CLASSIFICATION
# =========================

st.header("2. Engagement Behavior Classification")

posts = st.number_input("Posts per day", min_value=0, step=1)
likes = st.number_input("Likes per day", min_value=0, step=1)
follows = st.number_input("Follows per day", min_value=0, step=1)
app = st.selectbox("Platform", ["Instagram", "Facebook", "Twitter", "Snapchat"])

if st.button("Analyze Engagement"):
    payload = {
        "Posts_Per_Day": posts,
        "Likes_Per_Day": likes,
        "Follows_Per_Day": follows,
        "App": app
    }

    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Behavior Classification: {result['prediction']}")
        st.info(f"Confidence Score: {result['confidence_score']}")
    else:
        st.error("Prediction failed.")

st.divider()

# =========================
# SECTION 3: EMOTIONAL ANALYSIS
# =========================

st.header("3. Emotional Intelligence Analysis")

text_input = st.text_area("Enter social media text for analysis")

colA, colB, colC = st.columns(3)

if colA.button("Analyze Sentiment"):
    response = requests.post(f"{API_URL}/sentiment", json={"text": text_input})
    if response.status_code == 200:
        result = response.json()
        st.success(f"Sentiment: {result['label']}")
        st.info(f"Confidence: {result['score']}")

if colB.button("Detect Mood"):
    response = requests.post(f"{API_URL}/mood", json={"text": text_input})
    if response.status_code == 200:
        result = response.json()
        st.success(f"Mood: {result['mood']}")
        st.info(f"Confidence: {result['confidence']}")

if colC.button("Check Toxicity"):
    response = requests.post(f"{API_URL}/toxicity", json={"text": text_input})
    if response.status_code == 200:
        result = response.json()
        st.warning(f"Toxicity: {result['toxicity']}")
        st.info(f"Confidence: {result['confidence']}")

st.divider()

# =========================
# SECTION 4: TOPIC CLUSTERING
# =========================

st.header("4. Semantic Topic Clustering")

multi_text = st.text_area(
    "Enter multiple posts (one per line) for topic clustering"
)

if st.button("Cluster Topics"):
    texts = [t.strip() for t in multi_text.split("\n") if t.strip()]

    if texts:
        response = requests.post(
            f"{API_URL}/cluster",
            json={"texts": texts, "n_clusters": 3}
        )

        if response.status_code == 200:
            result = response.json()
            clusters = result["clusters"]

            df = pd.DataFrame({
                "Text": texts,
                "Cluster": clusters
            })

            st.dataframe(df)

            st.subheader("Cluster Distribution")
            st.bar_chart(df["Cluster"].value_counts())
    else:
        st.warning("Please enter at least one post.")

st.divider()

# =========================
# SECTION 5: HISTORY
# =========================

st.header("5. Privacy-Safe Prediction History")

history_response = requests.get(f"{API_URL}/history")

if history_response.status_code == 200:
    history = history_response.json()
    if history:
        df_history = pd.DataFrame(history)
        st.dataframe(df_history)
        st.subheader("Engagement Classification Distribution")
        st.bar_chart(df_history["prediction"].value_counts())
    else:
        st.write("No stored predictions yet.")
else:
    st.error("Unable to fetch history.")
