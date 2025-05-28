# 🤖 AI Text Summarizer

An intelligent text summarization application powered by Google's PEGASUS model and fine-tuned for dialogue summarization. This MLOps project demonstrates end-to-end machine learning pipeline implementation with modern deployment practices.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.20+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Live Demo

Try the application live at: **[https://text-summarizer-ai.onrender.com](https://text-summarizer-ai.onrender.com)**

⚠️ **Note**: The live demo is deployed on a free hosting platform and may experience errors or downtime. For the best experience and full functionality, we recommend running the application locally following the installation instructions below.

![image](https://github.com/user-attachments/assets/3dc576a2-cb41-4b2c-9a86-83c86a392c61)


## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Training Pipeline](#-training-pipeline)
- [Evaluation Metrics](#-evaluation-metrics)
- [Technologies Used](#-technologies-used)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a complete MLOps pipeline for text summarization using state-of-the-art transformer models. The application takes lengthy text inputs and generates concise, coherent summaries while maintaining the essential information and context.

### Key Highlights:
- **Fine-tuned PEGASUS Model**: Specialized for dialogue summarization
- **End-to-End MLOps Pipeline**: From data ingestion to deployment
- **Modern Web Interface**: Beautiful, responsive UI for easy interaction
- **RESTful API**: FastAPI-based backend for integration
- **Docker Containerized**: Ready for cloud deployment
- **Comprehensive Evaluation**: ROUGE metrics for model performance

## ✨ Features

- 🎯 **Intelligent Summarization**: Advanced NLP model for high-quality summaries
- 🌐 **Web Interface**: User-friendly interface with real-time character counting
- 🚀 **Fast API**: High-performance backend with automatic documentation
- 📊 **Model Evaluation**: Comprehensive metrics using ROUGE scores
- 🔄 **Training Pipeline**: Automated ML pipeline with configurable parameters
- 🐳 **Containerized**: Docker support for easy deployment
- 📱 **Responsive Design**: Works seamlessly across devices
- 🔧 **Modular Architecture**: Clean, maintainable codebase

## 🧠 Model Details

### Base Model
- **Model**: [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)
- **Architecture**: PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence)
- **Developer**: Google Research

### Fine-tuning Details
- **Dataset**: SAMSum Corpus (Samsung conversation dataset)
- **Task**: Dialogue summarization
- **Training Epochs**: 1 (optimized for efficiency)
- **Batch Size**: 1 per device
- **Learning Rate**: Adaptive with warmup
- **Evaluation Strategy**: Step-based evaluation every 500 steps

### Model Performance
The model is evaluated using ROUGE metrics:
- **ROUGE-1**: Measures unigram overlap
- **ROUGE-2**: Measures bigram overlap  
- **ROUGE-L**: Measures longest common subsequence
- **ROUGE-Lsum**: Summary-level ROUGE-L

## 📁 Project Structure

```
Text_Summarizer_AI/
├── 📊 artifacts/                      # Training artifacts and model outputs
│   ├── data_ingestion/               # Raw and processed datasets
│   ├── data_transformation/          # Tokenized datasets
│   ├── model_trainer/               # Trained models and tokenizers
│   └── model_evaluation/            # Evaluation metrics and results
├── 🐳 app.py                         # FastAPI application entry point
├── 🔧 config/
│   └── config.yaml                  # Configuration parameters
├── 🐳 Dockerfile                     # Container configuration
├── 📄 LICENSE                       # Project license
├── 🚀 main.py                       # Training pipeline orchestrator
├── ⚙️ params.yaml                    # Training hyperparameters
├── 📚 requirements.txt              # Python dependencies
├── 🔧 setup.py                      # Package installation script
├── 🔄 template.py                   # Project structure generator
├── 📔 research/                     # Jupyter notebooks for experimentation
│   ├── 1_data_ingestion.ipynb      # Data loading and preprocessing
│   ├── 2_data_transformation.ipynb  # Data tokenization experiments
│   ├── 3_model_trainer.ipynb       # Model training experiments
│   ├── 4_model_evaluation.ipynb    # Model evaluation experiments
│   └── research.ipynb              # General research notebook
├── 📦 src/textSummarizer/           # Main source code package
│   ├── components/                  # Core ML components
│   │   ├── data_ingestion.py       # Data download and extraction
│   │   ├── data_transformation.py   # Data preprocessing and tokenization
│   │   ├── model_trainer.py        # Model fine-tuning logic
│   │   └── model_evaluation.py     # Model evaluation and metrics
│   ├── config/                     # Configuration management
│   │   └── configuration.py        # Config loading and validation
│   ├── entity/                     # Data classes and entities
│   │   └── __init__.py             # Configuration entities
│   ├── pipeline/                   # ML pipelines
│   │   ├── stage_1_data_ingestion_pipeline.py
│   │   ├── stage_2_data_transformation_pipeline.py
│   │   ├── stage_3_model_trainer_pipeline.py
│   │   ├── stage_4_model_evaluation.py
│   │   └── prediction_pipeline.py   # Inference pipeline
│   ├── logging.py                  # Logging configuration
│   └── utils/                      # Utility functions
├── 🎨 static/                       # Frontend assets
│   ├── css/                        # Stylesheets
│   └── js/                         # JavaScript files
└── 🖥️ templates/                    # HTML templates
    └── index.html                  # Main web interface
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Git
- 4GB+ RAM (for model training)
- CUDA-compatible GPU (optional, for faster training)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Text_Summarizer_AI.git
   cd Text_Summarizer_AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t text-summarizer-ai .
   ```

2. **Run the container**
   ```bash
   docker run -p 8080:8080 text-summarizer-ai
   ```

## 💻 Usage

### Web Interface

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8080`

3. **Enter your text** in the input area and click "Generate Summary"

### API Usage

#### Generate Summary
```python
import requests

url = "http://localhost:8080/predict"
data = {"text": "Your long text here..."}

response = requests.post(url, json=data)
summary = response.json()["prediction"]
print(summary)
```

#### Training the Model
```python
import requests

url = "http://localhost:8080/train"
response = requests.get(url)
print(response.text)
```

### Command Line Training

Run the complete training pipeline:
```bash
python main.py
```

## 📖 API Documentation

FastAPI provides automatic API documentation:

- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

### Endpoints

#### `POST /predict`
Generate text summary
- **Request Body**: `{"text": "string"}`
- **Response**: `{"prediction": "summary"}`

#### `GET /predict`
Generate summary via query parameter
- **Query Parameter**: `text=your_text_here`
- **Response**: `{"prediction": "summary"}`

#### `GET /train`
Trigger model training
- **Response**: Training status message

## 🔄 Training Pipeline

The training pipeline consists of four main stages:

### Stage 1: Data Ingestion
- Downloads the SAMSum dataset
- Extracts and organizes data files
- Validates data integrity

### Stage 2: Data Transformation  
- Tokenizes dialogue and summary pairs
- Creates train/validation/test splits
- Prepares data for model training

### Stage 3: Model Training
- Loads pre-trained PEGASUS model
- Fine-tunes on dialogue summarization task
- Saves trained model and tokenizer

### Stage 4: Model Evaluation
- Evaluates model performance
- Calculates ROUGE metrics
- Generates evaluation report

### Configuration

Training parameters can be modified in `params.yaml`:

```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1e6
  gradient_accumulation_steps: 16
```

## 📊 Evaluation Metrics

The model is evaluated using standard summarization metrics:

- **ROUGE-1**: Measures the overlap of unigrams (single words)
- **ROUGE-2**: Measures the overlap of bigrams (two consecutive words)
- **ROUGE-L**: Measures the longest common subsequence
- **ROUGE-Lsum**: Summary-level ROUGE-L score

Results are saved in `artifacts/model_evaluation/metrics.csv`

## 🛠️ Technologies Used

### Machine Learning & NLP
- **🤗 Transformers**: State-of-the-art NLP models
- **PyTorch**: Deep learning framework
- **Datasets**: Efficient data loading and processing
- **Evaluate**: Model evaluation metrics
- **NLTK**: Natural language processing utilities

### Web Framework & API
- **FastAPI**: Modern, fast web framework
- **Uvicorn**: ASGI server
- **Jinja2**: Template engine
- **Pydantic**: Data validation

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **PyYAML**: YAML configuration files
- **python-box**: Advanced dictionary access

### DevOps & Deployment
- **Docker**: Containerization
- **Render**: Cloud deployment platform
- **Git**: Version control

## 🚀 Deployment

### Local Deployment
The application runs on `http://localhost:8080` by default.

### Cloud Deployment
Deployed on Render cloud platform with automatic CI/CD:
- **Live URL**: https://text-summarizer-ai.onrender.com
- **Container**: Docker-based deployment
- **Auto-scaling**: Handles variable traffic loads

### Docker Deployment
```bash
# Build
docker build -t text-summarizer .

# Run
docker run -p 8080:8080 text-summarizer
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

<div align="center">

**[🌟 Star this repo](https://github.com/yourusername/Text_Summarizer_AI)** if you found it helpful!

</div>
