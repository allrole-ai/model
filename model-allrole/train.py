import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling


# Langkah 1: Persiapan Data
df = pd.read_csv('qa.csv', delimiter='|', on_bad_lines='skip')

# Ganti nama kolom jika perlu
df.columns = ['question', 'answer']


# Load the dataset
def filter_valid_rows(row):
    return len(row) == 2

# Verifikasi kolom
print("Nama kolom dalam DataFrame:", df.columns)
print("Beberapa baris data:")
print(df.head())

# Pastikan kolom 'question' dan 'answer' adalah string
df['question'] = df['question'].astype(str)
df['answer'] = df['answer'].astype(str)

# Langkah 2: Tokenisasi dan Pembuatan Model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Menambahkan token padding ke tokenizer
tokenizer.pad_token = tokenizer.eos_token

class CustomDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=128):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

    inputs = self.tokenizer(question, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
    outputs = self.tokenizer(answer, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)

    input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension
    attention_mask = inputs['attention_mask'].squeeze(0)
    labels = outputs['input_ids'].squeeze(0)
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Buat dataset
dataset = CustomDataset(df['question'].tolist(), df['answer'].tolist(), tokenizer)

# Bagi dataset menjadi train dan eval
train_size = int(0.8 * len(dataset))





# add The training argument
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# add Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# make Training the model
trainer.train()

#Saves the model & tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

#Example of use for predictions with pre trained models
def predict(question, answer):
    inputs = tokenizer(
        question,
        answer,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

#Example using predictions
question = "cara minta transkrip nilai"
answer = "Yth. Kepala Bagian Akademik Universitas XYZdi TempatDengan hormat,Mahasiswa Universitas XYZ, Nama saya [Nama], dengan NIM [NIM]. Saya ingin meminta transkrip nilai semester [semester] yang telah saya tempuh.Demikian surat permohonan ini saya sampaikan, atas perhatian dan kerjasamanya saya ucapkan terima kasih.Hormat saya [Nama]"

predicted_class = predict(question, answer)
print(f"Pertanyaan: {question}")
print(f"Jawaban: {answer}")
print(f"Kelas Prediksi: {predicted_class}")

