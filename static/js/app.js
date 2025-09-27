// Plagiarism Checker JavaScript

class PlagiarismChecker {
    constructor() {
        this.fileInput = document.getElementById('fileInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.textInput = document.getElementById('textInput');
        this.checkButton = document.getElementById('checkPlagiarismBtn');
        this.clearButton = document.getElementById('clearBtn');
        this.progressSection = document.getElementById('progressSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        this.fileInfo = document.getElementById('fileInfo');
        
        this.currentFileId = null;
        this.isProcessing = false;
        
        this.initializeEventListeners();
        this.updateCharCount();
    }
    
    initializeEventListeners() {
        // File upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Text input events
        this.textInput.addEventListener('input', this.handleTextInput.bind(this));
        
        // Button events
        this.checkButton.addEventListener('click', this.checkPlagiarism.bind(this));
        this.clearButton.addEventListener('click', this.clearAll.bind(this));
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
    }
    
    handleDragOver(e) {
        this.uploadArea.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        this.uploadArea.classList.remove('dragover');
    }
    
    handleDrop(e) {
        this.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.fileInput.files = files;
            this.handleFileSelect();
        }
    }
    
    handleFileSelect() {
        const file = this.fileInput.files[0];
        if (!file) return;
        
        // Clear text input when file is selected
        this.textInput.value = '';
        this.updateCharCount();
        
        this.uploadFile(file);
    }
    
    handleTextInput() {
        // Clear file selection when text is entered
        if (this.textInput.value.trim()) {
            this.fileInput.value = '';
            this.currentFileId = null;
            this.hideFileInfo();
        }
        
        this.updateCharCount();
        this.updateButtonState();
    }
    
    updateCharCount() {
        const text = this.textInput.value;
        document.getElementById('charCount').textContent = text.length;
        document.getElementById('wordCount').textContent = text.trim().split(/\s+/).filter(word => word).length;
    }
    
    updateButtonState() {
        const hasFile = this.currentFileId !== null;
        const hasText = this.textInput.value.trim().length >= 50;
        this.checkButton.disabled = this.isProcessing || (!hasFile && !hasText);
    }
    
    async uploadFile(file) {
        try {
            this.showProgress('Uploading file...');
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                this.currentFileId = result.file_id;
                this.showFileInfo(result);
                this.updateButtonState();
                this.hideProgress();
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            this.showError('File upload failed: ' + error.message);
        }
    }
    
    async checkPlagiarism() {
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        this.updateButtonState();
        this.hideError();
        this.hideResults();
        
        try {
            let endpoint, data;
            
            if (this.currentFileId) {
                // Check file
                endpoint = '/check_plagiarism';
                data = { file_id: this.currentFileId };
                this.showProgress('Analyzing document for plagiarism...');
            } else {
                // Check text
                endpoint = '/check_text';
                data = { text: this.textInput.value.trim() };
                this.showProgress('Analyzing text for plagiarism...');
            }
            
            this.simulateProgress();
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                this.showResults(result.results);
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            this.showError('Plagiarism check failed: ' + error.message);
        } finally {
            this.isProcessing = false;
            this.updateButtonState();
            this.hideProgress();
        }
    }
    
    simulateProgress() {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const progressSection = this.progressSection.querySelector('.card-body');
        
        let progress = 0;
        let stage = 0;
        const stages = [
            'Preparing text for analysis...',
            'Creating optimized search chunks...',
            'Searching multiple engines in parallel...',
            'Fetching and analyzing content...',
            'Calculating similarity scores...',
            'Generating detailed report...'
        ];
        
        // Add percentage display
        if (!progressSection.querySelector('.progress-percentage')) {
            const percentDiv = document.createElement('div');
            percentDiv.className = 'progress-percentage';
            percentDiv.id = 'progressPercentage';
            progressSection.insertBefore(percentDiv, progressSection.querySelector('.progress'));
        }
        
        const interval = setInterval(() => {
            progress += Math.random() * 8 + 2; // Faster progress
            if (progress > 95) progress = 95;
            
            progressBar.style.width = progress + '%';
            document.getElementById('progressPercentage').textContent = `${Math.round(progress)}% Complete`;
            
            // Update stage
            const newStage = Math.floor(progress / 16);
            if (newStage !== stage && newStage < stages.length) {
                stage = newStage;
                progressText.textContent = stages[stage];
            }
        }, 400);
        
        // Clear interval when done
        setTimeout(() => {
            clearInterval(interval);
            progressBar.style.width = '100%';
            document.getElementById('progressPercentage').textContent = '100% Complete';
            progressText.textContent = 'Finalizing results...';
        }, 8000); // Shorter time for faster processing
    }
    
    showFileInfo(fileData) {
        const fileDetails = document.getElementById('fileDetails');
        fileDetails.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <strong>Filename:</strong> ${fileData.filename}<br>
                    <strong>Words:</strong> ${fileData.word_count.toLocaleString()}
                </div>
                <div class="col-md-6">
                    <strong>Characters:</strong> ${fileData.character_count.toLocaleString()}<br>
                    <small class="text-muted">Ready for analysis</small>
                </div>
            </div>
        `;
        this.fileInfo.style.display = 'block';
    }
    
    hideFileInfo() {
        this.fileInfo.style.display = 'none';
    }
    
    showProgress(message) {
        document.getElementById('progressText').textContent = message;
        document.getElementById('progressBar').style.width = '0%';
        this.progressSection.style.display = 'block';
        
        // Add warning about not refreshing
        const progressBody = this.progressSection.querySelector('.card-body');
        if (!progressBody.querySelector('.progress-warning')) {
            const warningDiv = document.createElement('div');
            warningDiv.className = 'progress-warning';
            warningDiv.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Please wait:</strong> Analysis may take 1-3 minutes for longer documents. 
                Do not refresh or close this tab during processing.
            `;
            progressBody.appendChild(warningDiv);
        }
    }
    
    hideProgress() {
        this.progressSection.style.display = 'none';
    }
    
    showResults(results) {
        const resultsBody = this.resultsSection.querySelector('.card-body');
        
        // Determine score class
        let scoreClass = 'score-low';
        if (results.plagiarism_percentage > 15) scoreClass = 'score-medium';
        if (results.plagiarism_percentage > 30) scoreClass = 'score-high';
        if (results.plagiarism_percentage > 50) scoreClass = 'score-critical';
        
        let html = `
            <div class="plagiarism-score-container">
                <div class="plagiarism-score ${scoreClass}">
                    ${results.plagiarism_percentage}%
                </div>
                <h6>Plagiarism Detected</h6>
            </div>
            
            <div class="row results-stats-row">
                <div class="col-6 col-md-3">
                    <div class="stat-card">
                        <span class="stat-value">${results.total_words.toLocaleString()}</span>
                        <span class="stat-label">Total Words</span>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="stat-card">
                        <span class="stat-value">${results.plagiarized_words.toLocaleString()}</span>
                        <span class="stat-label">Flagged Words</span>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="stat-card">
                        <span class="stat-value">${results.sources_found}</span>
                        <span class="stat-label">Sources Found</span>
                    </div>
                </div>
                <div class="col-6 col-md-3">
                    <div class="stat-card">
                        <span class="stat-value">${results.matches ? results.matches.length : 0}</span>
                        <span class="stat-label">Matches</span>
                    </div>
                </div>
            </div>
        `;
        
        // Analysis summary
        html += `
            <div class="alert alert-info">
                <h6><i class="fas fa-info-circle me-2"></i>Analysis Summary</h6>
                <p class="mb-0">${results.analysis_summary}</p>
            </div>
        `;
        
        // Sources
        if (results.sources && results.sources.length > 0) {
            html += `<h6 class="mt-4 mb-3"><i class="fas fa-external-link-alt me-2"></i>Sources Found</h6>`;
            
            results.sources.forEach((source, index) => {
                html += `
                    <div class="source-card p-3">
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <h6>
                                    <a href="${source.url}" target="_blank" class="source-title">
                                        ${source.title || 'Untitled Source'}
                                    </a>
                                </h6>
                                <small class="text-muted">${source.url}</small>
                                <div class="mt-2">
                                    <span class="badge bg-primary">
                                        ${source.match_count} match${source.match_count !== 1 ? 'es' : ''}
                                    </span>
                                    <span class="badge bg-warning ms-1">
                                        ${(source.avg_similarity * 100).toFixed(1)}% similarity
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
        }
        
        // Individual matches
        if (results.matches && results.matches.length > 0) {
            html += `<h6 class="mt-4 mb-3"><i class="fas fa-list me-2"></i>Detailed Matches</h6>`;
            
            results.matches.slice(0, 10).forEach((match, index) => {
                html += `
                    <div class="match-card card p-3 mb-3">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <h6 class="mb-0">Match ${index + 1}</h6>
                            <span class="badge bg-warning similarity-badge">
                                ${(match.similarity * 100).toFixed(1)}% Similar
                            </span>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <strong>Original Text:</strong>
                                <div class="matched-text">${match.original_text}</div>
                            </div>
                            <div class="col-md-6">
                                <strong>Found in Source:</strong>
                                <div class="matched-text">${match.matched_text}</div>
                                <small class="text-muted mt-1">
                                    <a href="${match.url}" target="_blank">${match.title}</a>
                                </small>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            if (results.matches.length > 10) {
                html += `<p class="text-muted">Showing top 10 matches. ${results.matches.length - 10} more matches found.</p>`;
            }
        }
        
        resultsBody.innerHTML = html;
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideResults() {
        this.resultsSection.style.display = 'none';
    }
    
    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        this.errorSection.style.display = 'block';
        this.errorSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideError() {
        this.errorSection.style.display = 'none';
    }
    
    clearAll() {
        this.fileInput.value = '';
        this.textInput.value = '';
        this.currentFileId = null;
        this.updateCharCount();
        this.updateButtonState();
        this.hideFileInfo();
        this.hideResults();
        this.hideError();
        this.hideProgress();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PlagiarismChecker();
});