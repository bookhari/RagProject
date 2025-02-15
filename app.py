from flask import Flask, request, jsonify, send_file
import os
import time
from werkzeug.utils import secure_filename
from backend import embed_and_store_text, ask_chatgpt_arabic

app = Flask(__name__)

# Serve the index.html file when accessing the root URL.
@app.route('/')
def index():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    
    try:
        embed_and_store_text(filepath)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'message': 'File processed successfully!'}), 200

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query', '')
    if not query_text:
        return jsonify({'error': 'Query text is required.'}), 400

    try:
        answer = ask_chatgpt_arabic(query_text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'answer': answer}), 200

if __name__ == '__main__':
    app.run(debug=True)
