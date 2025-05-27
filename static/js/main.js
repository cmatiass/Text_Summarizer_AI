// Modern Text Summarizer JavaScript

class TextSummarizer {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.updateCharCount();
    }

    initializeElements() {
        this.inputText = document.getElementById('inputText');
        this.charCount = document.getElementById('charCount');
        this.summarizeBtn = document.getElementById('summarizeBtn');
        this.summaryOutput = document.getElementById('summaryOutput');
        this.summaryActions = document.getElementById('summaryActions');
        this.copyBtn = document.getElementById('copyBtn');
        this.compressionRatio = document.getElementById('compressionRatio');
        this.loadingIndicator = document.getElementById('loadingIndicator');
    }

    attachEventListeners() {
        // Input text events
        this.inputText.addEventListener('input', () => this.updateCharCount());
        this.inputText.addEventListener('paste', () => {
            setTimeout(() => this.updateCharCount(), 10);
        });

        // Summarize button
        this.summarizeBtn.addEventListener('click', () => this.generateSummary());

        // Copy button
        this.copyBtn.addEventListener('click', () => this.copySummary());

        // Enter key in textarea (Ctrl+Enter to summarize)
        this.inputText.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.generateSummary();
            }
        });
    }

    updateCharCount() {
        const text = this.inputText.value;
        const charLength = text.length;
        this.charCount.textContent = charLength.toLocaleString();
        
        // Update button state
        const minLength = 50;
        if (charLength < minLength) {
            this.summarizeBtn.disabled = true;
            this.summarizeBtn.innerHTML = `<i class="fas fa-magic"></i> Need ${minLength - charLength} more characters`;
        } else {
            this.summarizeBtn.disabled = false;
            this.summarizeBtn.innerHTML = '<i class="fas fa-magic"></i> Generate Summary';
        }
    }

    async generateSummary() {
        const text = this.inputText.value.trim();
        
        if (!text || text.length < 50) {
            this.showAlert('Please enter at least 50 characters for better summarization results.', 'error');
            return;
        }

        try {
            this.showLoading(true);
            this.updateButtonState('loading');
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displaySummary(data.prediction, text);
            
        } catch (error) {
            console.error('Error generating summary:', error);
            this.showAlert('Failed to generate summary. Please try again later.', 'error');
        } finally {
            this.showLoading(false);
            this.updateButtonState('normal');
        }
    }

    displaySummary(summary, originalText) {
        // Clear placeholder
        this.summaryOutput.innerHTML = '';
        this.summaryOutput.classList.add('has-content');

        // Create summary content
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'summary-content';
        summaryDiv.textContent = summary;

        this.summaryOutput.appendChild(summaryDiv);

        // Show summary actions
        this.summaryActions.style.display = 'flex';

        // Calculate and display compression ratio
        const originalLength = originalText.length;
        const summaryLength = summary.length;
        const compressionPercentage = Math.round((1 - summaryLength / originalLength) * 100);
        
        this.compressionRatio.innerHTML = `
            <i class="fas fa-chart-line"></i>
            ${compressionPercentage}% compression 
            (${originalLength.toLocaleString()} → ${summaryLength.toLocaleString()} chars)
        `;

        // Show success message
        this.showAlert('Summary generated successfully!', 'success');

        // Smooth scroll to summary
        this.summaryOutput.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
    }

    async copySummary() {
        const summaryContent = this.summaryOutput.querySelector('.summary-content');
        if (!summaryContent) return;

        try {
            await navigator.clipboard.writeText(summaryContent.textContent);
            
            // Update button text temporarily
            const originalHTML = this.copyBtn.innerHTML;
            this.copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            this.copyBtn.style.background = 'var(--success-color)';
            this.copyBtn.style.color = 'white';
            
            setTimeout(() => {
                this.copyBtn.innerHTML = originalHTML;
                this.copyBtn.style.background = '';
                this.copyBtn.style.color = '';
            }, 2000);
            
        } catch (error) {
            console.error('Failed to copy text:', error);
            this.showAlert('Failed to copy summary to clipboard.', 'error');
        }
    }

    showLoading(show) {
        this.loadingIndicator.style.display = show ? 'flex' : 'none';
    }

    updateButtonState(state) {
        switch (state) {
            case 'loading':
                this.summarizeBtn.disabled = true;
                this.summarizeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
                break;
            case 'normal':
                this.summarizeBtn.disabled = false;
                this.summarizeBtn.innerHTML = '<i class="fas fa-magic"></i> Generate Summary';
                break;
        }
    }

    showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert');
        existingAlerts.forEach(alert => alert.remove());

        // Create new alert
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
            ${message}
        `;

        // Insert after the input section
        const inputSection = document.querySelector('.input-section');
        inputSection.insertAdjacentElement('afterend', alert);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
}

// Utility Functions
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Enhanced Textarea Features
function enhanceTextarea() {
    const textarea = document.getElementById('inputText');
    
    // Auto-resize functionality
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 400) + 'px';
    });

    // Keyboard shortcuts
    textarea.addEventListener('keydown', function(e) {
        // Tab key indentation
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = this.selectionStart;
            const end = this.selectionEnd;
            this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
            this.selectionStart = this.selectionEnd = start + 4;
        }
    });
}

// Sample text functionality
function loadSampleText() {
    const sampleTexts = [
        `Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, revolutionizing industries from healthcare to finance. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with unprecedented accuracy. Deep learning models, inspired by the structure of the human brain, have achieved remarkable breakthroughs in image recognition, natural language processing, and game playing. However, the rapid advancement of AI also raises important ethical questions about privacy, job displacement, and the need for responsible development. As we continue to integrate AI into our daily lives, it's crucial to balance innovation with careful consideration of its societal impacts.`,
        
        `Climate change represents one of the most pressing challenges facing humanity today. Rising global temperatures, caused primarily by increased greenhouse gas emissions from human activities, are leading to more frequent extreme weather events, rising sea levels, and shifts in precipitation patterns. These changes threaten ecosystems, food security, and human settlements worldwide. Addressing climate change requires immediate action on multiple fronts: transitioning to renewable energy sources, improving energy efficiency, protecting and restoring natural carbon sinks like forests, and developing new technologies for carbon capture and storage. International cooperation and commitment from governments, businesses, and individuals are essential to limit global warming and build resilience against its impacts.`,
        
        `The digital revolution has fundamentally transformed how we communicate, work, and access information. The internet has connected billions of people worldwide, enabling instant communication and collaboration across vast distances. Social media platforms have changed how we share experiences and form communities, while e-commerce has revolutionized retail and business models. Cloud computing has made powerful computing resources accessible to organizations of all sizes, fostering innovation and entrepreneurship. However, this digital transformation has also brought challenges including cybersecurity threats, privacy concerns, digital divides, and the spread of misinformation. As we become increasingly dependent on digital technologies, ensuring their security, accessibility, and responsible use becomes paramount.`
    ];

    const randomText = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
    const textarea = document.getElementById('inputText');
    textarea.value = randomText;
    
    // Trigger events to update UI
    textarea.dispatchEvent(new Event('input'));
    
    // Smooth scroll to textarea
    textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    textarea.focus();
}

// Add sample text button functionality
function addSampleTextButton() {
    const inputActions = document.querySelector('.input-actions');
    const sampleBtn = document.createElement('button');
    sampleBtn.type = 'button';
    sampleBtn.className = 'btn-secondary';
    sampleBtn.innerHTML = '<i class="fas fa-file-text"></i> Load Sample';
    sampleBtn.onclick = loadSampleText;
    
    inputActions.insertBefore(sampleBtn, inputActions.lastElementChild);
}

// Performance monitoring
function initPerformanceMonitoring() {
    // Monitor API response times
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        const startTime = performance.now();
        return originalFetch.apply(this, args)
            .then(response => {
                const endTime = performance.now();
                const duration = endTime - startTime;
                console.log(`API call took ${duration.toFixed(2)}ms`);
                return response;
            });
    };
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize main application
    new TextSummarizer();
    
    // Initialize enhanced features
    enhanceTextarea();
    addSampleTextButton();
    initPerformanceMonitoring();
    
    // Add smooth scrolling to all internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading animation for better UX
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.3s ease-in-out';
    
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

// Error handling for uncaught errors
window.addEventListener('error', function(e) {
    console.error('Uncaught error:', e.error);
    // You could send this to a logging service in production
});

// Service worker registration for potential PWA features
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Uncomment when you add a service worker
        // navigator.serviceWorker.register('/sw.js');
    });
}
