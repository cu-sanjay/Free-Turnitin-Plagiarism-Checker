const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const PlagiarismChecker = require('./lib/plagiarism-checker');
const TextExtractor = require('./lib/text-extractor');

const app = express();
const port = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static('public'));

// Configure multer for file uploads
const uploadDir = './uploads';
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ 
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
    fileFilter: (req, file, cb) => {
        const allowedTypes = /\.(txt|pdf|docx)$/i;
        if (allowedTypes.test(file.originalname)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only TXT, PDF, and DOCX files are allowed.'));
        }
    }
});

// Initialize services
const plagiarismChecker = new PlagiarismChecker();
const textExtractor = new TextExtractor();

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', service: 'plagiarism-checker-node' });
});

app.post('/api/upload', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        console.log('Processing uploaded file:', req.file.originalname);
        
        // Extract text from file
        const extractedText = await textExtractor.extractText(req.file.path);
        
        if (!extractedText || extractedText.trim().length < 50) {
            fs.unlinkSync(req.file.path); // Clean up
            return res.status(400).json({ error: 'Could not extract sufficient text from the file' });
        }

        // Generate file ID for session tracking
        const fileId = path.basename(req.file.filename, path.extname(req.file.filename));

        res.json({
            success: true,
            fileId: fileId,
            filename: req.file.originalname,
            textPreview: extractedText.substring(0, 500) + (extractedText.length > 500 ? '...' : ''),
            wordCount: extractedText.split(/\s+/).filter(word => word.length > 0).length,
            characterCount: extractedText.length,
            filePath: req.file.path
        });

    } catch (error) {
        console.error('Upload error:', error);
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        res.status(500).json({ error: 'Error processing file: ' + error.message });
    }
});

app.post('/api/check-plagiarism', async (req, res) => {
    try {
        const { fileId } = req.body;
        
        if (!fileId) {
            return res.status(400).json({ error: 'File ID is required' });
        }

        // Find the uploaded file
        const uploadedFiles = fs.readdirSync(uploadDir).filter(file => file.includes(fileId));
        
        if (uploadedFiles.length === 0) {
            return res.status(404).json({ error: 'File not found' });
        }

        const filePath = path.join(uploadDir, uploadedFiles[0]);
        const text = await textExtractor.extractText(filePath);
        
        console.log('Checking plagiarism for file, text length:', text.length);
        
        // Perform plagiarism check
        const results = await plagiarismChecker.checkPlagiarism(text);
        
        // Clean up file
        fs.unlinkSync(filePath);
        
        res.json({
            success: true,
            results: results
        });

    } catch (error) {
        console.error('Plagiarism check error:', error);
        res.status(500).json({ error: 'Error checking plagiarism: ' + error.message });
    }
});

app.post('/api/check-text', async (req, res) => {
    try {
        const { text } = req.body;
        
        if (!text || text.trim().length < 50) {
            return res.status(400).json({ error: 'Text must be at least 50 characters long' });
        }

        console.log('Checking plagiarism for text, length:', text.length);
        
        // Perform plagiarism check
        const results = await plagiarismChecker.checkPlagiarism(text);
        
        res.json({
            success: true,
            results: results
        });

    } catch (error) {
        console.error('Text check error:', error);
        res.status(500).json({ error: 'Error checking plagiarism: ' + error.message });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File too large. Maximum size is 10MB.' });
        }
    }
    console.error('Server error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(port, '0.0.0.0', () => {
    console.log(`Plagiarism Checker server running on port ${port}`);
});

module.exports = app;