from flask import Flask, render_template, request
from qa_model import get_response

app = Flask(__name__)

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint untuk menerima pertanyaan
@app.route('/ask', methods=['POST'])
def ask():
    if request.method == 'POST':
        question = request.form['question']
        answer = get_response(question)
        return answer

if __name__ == '__main__':
    app.run(debug=True)
