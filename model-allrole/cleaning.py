import csv
import re

# Daftar simbol yang akan dihapus
symbols_to_remove = [
    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "=", "[", "]",
    "{", "}", ";", ":", "'", "|", ",", ".", "<", ">", "/", "?", "\n"
]

# Membuat pola regex untuk menghapus simbol-simbol tersebut
pattern = re.compile(f"[{''.join(re.escape(symbol) for symbol in symbols_to_remove)}]")

def clean_text(text):
    """Membersihkan teks dari simbol-simbol yang tidak diinginkan."""
    return pattern.sub('', text)

input_file = 'dataset\dataset-a.csv'
output_file = 'dataset\qa-dataset.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile, delimiter='|')
    writer = csv.writer(outfile, delimiter='|')
    
    for row in reader:
        if len(row) == 2:
            cleaned_row = [clean_text(field) for field in row]
            writer.writerow(cleaned_row)
        else:
            print(f"Baris tidak valid: {row}")

print("File CSV telah dibersihkan dan disimpan sebagai 'qa_cleaned.csv'")
