import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Memuat vectorizer dan data yang telah disimpan
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/data.pkl', 'rb') as f:
    data = pickle.load(f)