






    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = QADataset(train_encodings, train_labels)
val_dataset = QADataset(val_encodings, val_labels)

# Set up Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch"  # Updated parameter name
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Melatih model
trainer.train()

# Evaluasi model
trainer.evaluate()

# Simpan model dan tokenizer
if not os.path.exists('model'):
    os.makedirs('model')
model.save_pretrained('model/qa_model')
tokenizer.save_pretrained('model/qa_tokenizer')

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