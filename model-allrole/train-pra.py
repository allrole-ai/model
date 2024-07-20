import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import pickle
import os
import re
from tqdm import tqdm




print("Akurasi: {:.2f}%".format(accuracy * 100))
print("Presisi: {:.2f}%".format(precision * 100))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# Membuat folder 'model' jika belum ada
if not os.path.exists('model'):
    os.makedirs('model')

# Simpan vectorizer, model, dan data ke dalam folder 'model'
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Model dan data telah disimpan ke dalam folder 'model'.")
