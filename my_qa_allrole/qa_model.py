import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Memuat vectorizer dan data yang sudah disimpan
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

