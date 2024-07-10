import csv
import re

# Daftar simbol yang akan dihapus
symbols_to_remove = [
    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "=", "[", "]",
    "{", "}", ";", ":", "'", "|", ",", ".", "<", ">", "/", "?", "\n"
]