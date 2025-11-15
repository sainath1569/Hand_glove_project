class GloveDefectApp {
    constructor() {
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const processBtn = document.getElementById('processBtn');
        const newAnalysisBtn = document.getElementById('newAnalysisBtn');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));

        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        processBtn.addEventListener('click', this.processImage.bind(this));
        newAnalysisBtn.addEventListener('click', this.resetAnalysis.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.add('border-blue-400', 'bg-blue-50');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('border-blue-400', 'bg-blue-50');
    }

    handleDrop(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('border-blue-400', 'bg-blue-50');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFiles(files[0]);
        }
    }

    handleFileSelect(e) {
        if (e.target.files.length > 0) {
            this.handleFiles(e.target.files[0]);
        }
    }

    handleFiles(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG, etc.)');
            return;
        }

        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return;
        }

        // Display file info
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.innerHTML = `
            <div class="text-4xl mb-4 text-green-500">✅</div>
            <h3 class="text-xl font-semibold mb-2">File Selected</h3>
            <p class="text-gray-600">${file.name}</p>
            <p class="text-sm text-gray-500">Click "Process Image" to continue</p>
        `;

        // Enable process button
        document.getElementById('processBtn').disabled = false;
        this.currentFile = file;
    }

    async processImage() {
        if (!this.currentFile) {
            alert('Please select an image file first.');
            return;
        }

        const loadingIndicator = document.getElementById('loadingIndicator');
        const processBtn = document.getElementById('processBtn');
        
        // Show loading
        loadingIndicator.classList.remove('hidden');
        processBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.displayResults(result.results);
            } else {
                throw new Error(result.message || 'Processing failed');
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Error processing image: ' + error.message);
        } finally {
            loadingIndicator.classList.add('hidden');
            processBtn.disabled = false;
        }
    }

    displayResults(results) {
        // Show results section
        document.getElementById('results').classList.remove('hidden');
        
        // Scroll to results
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

        // Update images
        document.getElementById('originalImage').src = results.original_image;
        document.getElementById('processedImage').src = results.detected_image || results.enhanced_image || results.original_image;

        // Update metrics
        document.getElementById('defectCount').textContent = results.total_defects || 0;
        document.getElementById('confidenceScore').textContent = this.calculateOverallConfidence(results.detections) + '%';
        
        // For demo purposes - in real implementation, these would come from the backend
        document.getElementById('precisionScore').textContent = '94%';
        document.getElementById('recallScore').textContent = '89%';

        // Update defects list
        this.updateDefectsList(results.detections || []);
    }

    calculateOverallConfidence(detections) {
        if (!detections || detections.length === 0) return 0;
        
        const totalConfidence = detections.reduce((sum, detection) => sum + detection.confidence, 0);
        return Math.round((totalConfidence / detections.length) * 100);
    }

    updateDefectsList(detections) {
        const container = document.getElementById('defectsContainer');
        
        if (detections.length === 0) {
            container.innerHTML = '<p class="text-gray-500 text-center py-4">No defects detected</p>';
            return;
        }

        container.innerHTML = detections.map(detection => `
            <div class="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                <div class="flex items-center space-x-3">
                    <div class="w-3 h-3 rounded-full ${this.getDefectColor(detection.class_name)}"></div>
                    <span class="font-medium">${detection.class_name}</span>
                </div>
                <div class="text-right">
                    <div class="font-bold text-blue-600">${Math.round(detection.confidence * 100)}%</div>
                    <div class="text-sm text-gray-500">confidence</div>
                </div>
            </div>
        `).join('');
    }

    getDefectColor(defectType) {
        const colors = {
            'large': 'bg-red-500',
            'medium': 'bg-orange-500',
            'small': 'bg-yellow-500'
        };
        return colors[defectType.toLowerCase()] || 'bg-gray-500';
    }

    resetAnalysis() {
        // Reset upload area
        document.getElementById('uploadArea').innerHTML = `
            <div class="text-4xl mb-4">📁</div>
            <h3 class="text-xl font-semibold mb-2">Drag & Drop or Click to Upload</h3>
            <p class="text-gray-500">Supported formats: JPG, PNG (Max 10MB)</p>
        `;
        
        // Reset file input
        document.getElementById('fileInput').value = '';
        document.getElementById('processBtn').disabled = true;
        this.currentFile = null;

        // Hide results
        document.getElementById('results').classList.add('hidden');

        // Scroll to upload section
        document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new GloveDefectApp();
});