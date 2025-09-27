from flask import Flask, render_template, request, jsonify, send_file
import os
import hashlib
from werkzeug.utils import secure_filename
from plagiarism_checker import PlagiarismChecker
from text_extractor import TextExtractor
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize plagiarism checker
plagiarism_checker = PlagiarismChecker()
text_extractor = TextExtractor()

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload TXT, PDF, or DOCX files only.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename or 'uploaded_file')
        file_id = hashlib.md5(f"{filename}{datetime.now()}".encode()).hexdigest()[:8]
        filename = f"{file_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from file
        extracted_text = text_extractor.extract_text(filepath)
        if not extracted_text.strip():
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from the file or file is empty'}), 400
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': file.filename,
            'text_preview': extracted_text[:500] + ('...' if len(extracted_text) > 500 else ''),
            'word_count': len(extracted_text.split()),
            'character_count': len(extracted_text)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/check_plagiarism', methods=['POST'])
def check_plagiarism():
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({'error': 'File ID is required'}), 400
        
        # Find the uploaded file
        uploaded_file = None
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(f"{file_id}_"):
                uploaded_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                break
        
        if not uploaded_file:
            return jsonify({'error': 'File not found'}), 404
        
        # Extract text
        text = text_extractor.extract_text(uploaded_file)
        
        # Check for plagiarism
        results = plagiarism_checker.check_plagiarism(text)
        
        # Clean up uploaded file
        os.remove(uploaded_file)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Error checking plagiarism: {str(e)}'}), 500

@app.route('/check_text', methods=['POST'])
def check_text():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if len(text) < 50:
            return jsonify({'error': 'Text must be at least 50 characters long for meaningful analysis'}), 400
        
        # Check for plagiarism
        results = plagiarism_checker.check_plagiarism(text)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Error checking plagiarism: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'plagiarism-checker'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)