"""
Pipeline Package for Text Summarizer AI

This package contains all the pipeline modules for the text summarization ML project.
Each stage represents a different phase of the machine learning workflow:

- stage_1_data_ingestion_pipeline.py: Downloads and extracts the training dataset
- stage_2_data_transformation_pipeline.py: Preprocesses and transforms raw data for training
- stage_3_model_trainer_pipeline.py: Handles the model training process
- stage_4_model_evaluation.py: Evaluates the trained model performance
- predicition_pipeline.py: Provides inference capabilities for generating summaries

These pipelines follow a modular design pattern, making the ML workflow
organized, maintainable, and easily executable in sequence or individually.
"""