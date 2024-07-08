import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('indobert_model')
model = TFBertForSequenceClassification.from_pretrained('indobert_model')

# Load the label encoder
with open('dataset/dataset-a.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])

# Function to make predictions
def predict(question):
    encoded = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=64,
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

# DataFrame to store questions and answers
qa_log = pd.DataFrame(columns=['question', 'answer'])

# Interactive loop to get user input and predict
while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    answer = predict(question)
    print(f"Predicted Answer: {answer}")
    # Add the question and answer to the log
    qa_log = pd.concat([qa_log, pd.DataFrame({'question': [question], 'answer': [answer]})], ignore_index=True)

# Save the log to an Excel file
qa_log.to_excel('qa_log.xlsx', index=False)
print("All questions and answers have been saved to 'qa_log.xlsx'.")
