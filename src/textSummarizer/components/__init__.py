"""
Components Package for Text Summarizer AI

This package contains the core component classes that implement the main functionality
of each stage in the text summarization ML pipeline. Each component is responsible
for a specific aspect of the machine learning workflow.

Component Overview:
- DataIngestion: Downloads and extracts datasets from external sources
- DataTransformation: Preprocesses and tokenizes text data for model training
- ModelTrainer: Fine-tunes pre-trained transformer models on domain-specific data
- ModelEvaluation: Evaluates trained models using ROUGE and other metrics

Key Design Principles:
- Each component is self-contained and focused on a single responsibility
- Components follow a consistent interface pattern with configuration-based initialization
- All components integrate with the centralized logging system for monitoring
- Components handle both GPU and CPU execution environments
- Error handling and resource management are built into each component

Architecture Benefits:
- Modular design enables independent testing and development of each stage
- Configuration-driven approach allows easy parameter tuning without code changes
- Clear separation of concerns makes the pipeline maintainable and extensible
- Components can be reused in different pipeline configurations or projects

Each component can be used independently or as part of the complete pipeline,
providing flexibility for different deployment scenarios and experimentation workflows.
"""