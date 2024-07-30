


























# Simpan model dan tokenizer
if not os.path.exists('model'):
    os.makedirs('model')


# Fungsi untuk menjawab pertanyaan baru
nlp = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True)

def answer_question(question, context=[]):
    """
    Menjawab pertanyaan berdasarkan konteks percakapan sebelumnya.
    
    Parameters:
    - question (str): Pertanyaan yang diajukan.
    - context (list): Daftar konteks percakapan sebelumnya.
    
    Returns:
    - str: Jawaban yang dihasilkan oleh model.
    """
    context.append(question)
    input_text = " ".join(context)
    result = nlp(input_text)
    label = max(result[0], key=lambda x: x['score'])['label']
    return id2label[int(label.split('_')[-1])]
