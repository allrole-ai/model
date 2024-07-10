import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

# Memuat data dari file yang sudah dibersihkan
data = pd.read_csv('dataset/qa-dataset.csv', delimiter='|')

# Membuat vectorizer dan menyesuaikan dengan data
vectorizer = CountVectorizer()
vectorizer.fit(data['question'])

# Membuat folder 'model' jika belum ada
if not os.path.exists('model'):
    os.makedirs('model')

# Simpan vectorizer dan data ke dalam folder 'model'
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model/data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Model dan data telah disimpan ke dalam folder 'model'.")
