import pandas as pd

# Coba baca file CSV dengan penanganan error
try:
    data = pd.read_csv('dataset\qa-dataset.csv', delimiter='|', on_bad_lines='warn')
    print("Data berhasil dibaca:")
    print(data.head())
except pd.errors.ParserError as e:
    print("Error membaca file CSV:", e)
