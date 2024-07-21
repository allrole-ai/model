import tensorflow as tf
from transformers import TFAlbertForSequenceClassification, AlbertTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load tokenizer dan model
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2')

# Load Encoder label
with open('rf1.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])

# function to make pedictions 
def predict(question):
    encoded = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    logits = model(input_ids, attention_mask=attention_mask).logits
    predicted_label_id = tf.argmax(logits, axis=1).numpy()[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_id])[0]
    
    return predicted_label

# Hitung akurasi Pada Set Tes
test_accuracy_metric = tf.keras.metrics.Accuracy()

