import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Memuat vectorizer dan data yang telah disimpan
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/data.pkl', 'rb') as f:
    data = pickle.load(f)


def get_response(query):
    query_vec = vectorizer.transform([query]).toarray()
    data_vecs = vectorizer.transform(data['question']).toarray()
    similarities = cosine_similarity(query_vec, data_vecs).flatten()
    closest_idx = np.argmax(similarities)
    return data['answer'].iloc[closest_idx]