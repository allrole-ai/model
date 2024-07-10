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