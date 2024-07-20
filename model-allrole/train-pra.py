import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
import pickle
import os
import re
from tqdm import tqdm


)

# Membuat vectorizer dan menyesuaikan dengan data
vectorizer = TfidfVectorizer()
print("Fitting vectorizer...")
X = vectorizer.fit_transform(tqdm(data['question'], desc="Fitting vectorizer"))

# Membagi data menjadi training dan testing
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, data['answer'], test_size=0.2, random_state=42)

# Menggunakan GridSearchCV untuk mencari hyperparameter terbaik
print("Training model...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Prediksi pada data test
print("Predicting...")
y_pred = best_model.predict(X_test)

# Menghitung akurasi dan presisi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')

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
