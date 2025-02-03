from flask import Flask, request, jsonify, render_template
import ollama
import os
import PyPDF2  

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

VALID_MODELS = ["llama3", "llama3.1"]

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def process_text_with_model(model, document_text, query):
    try:
    
        if model not in VALID_MODELS:
            return f"Error: Model '{model}' is not available. Please choose a valid model like 'llama3' or 'llama3.1'."
        
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': query + ' ' + document_text[:1000]}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error processing with model: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    query = request.form.get('query', '')  

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f'Error saving the file: {str(e)}'}), 500

    if file.filename.endswith('.pdf'):
        document_text = extract_text_from_pdf(file_path)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                document_text = f.read()
        except Exception as e:
            return jsonify({'error': f'Error reading the file: {str(e)}'}), 500

    model = request.form.get('model', 'llama3')
    if model not in VALID_MODELS:
        model = 'llama3'

    response = process_text_with_model(model, document_text, query)
    return render_template('index.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
