

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

# Loop untuk menerima pertanyaan terus-menerus
while True:
    pertanyaan = input("Masukkan pertanyaan (ketik 'exit' untuk keluar): ")
    if pertanyaan.lower() == 'exit':
        print("Terima kasih! Sampai jumpa.")
        break
    jawaban = get_response(pertanyaan)
    print(f"Jawaban: {jawaban}")