





























# Fungsi untuk menjawab pertanyaan baru
nlp = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

def answer_question(question, context=[]):
    """
    Menjawab pertanyaan berdasarkan konteks percakapan sebelumnya.
    
    Parameters:
    - question (str): Pertanyaan yang diajukan.
    - context (list): Daftar konteks percakapan sebelumnya.
    
