

# Loop untuk menerima pertanyaan terus-menerus
while True:
    pertanyaan = input("Masukkan pertanyaan (ketik 'exit' untuk keluar): ")
    if pertanyaan.lower() == 'exit':
        print("Terima kasih! Sampai jumpa.")
        break
    jawaban = get_response(pertanyaan)
    print(f"Jawaban: {jawaban}")