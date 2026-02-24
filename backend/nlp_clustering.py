from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import umap
# Load lightweight embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts: list):
    """
    Convert list of texts into sentence embeddings
    """
    embeddings = embedding_model.encode(texts)
    return np.array(embeddings)


def cluster_texts(texts: list, n_clusters: int = 3):
    """
    Perform KMeans clustering on sentence embeddings
    """
    embeddings = generate_embeddings(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    return {
        "clusters": labels.tolist()
    }


def cluster_texts_with_projection(texts: list, n_clusters: int = 3):
    embeddings = generate_embeddings(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    reducer = umap.UMAP(n_components=2, random_state=42)
    projection = reducer.fit_transform(embeddings)

    return {
        "clusters": labels.tolist(),
        "projection": projection.tolist()
    }
