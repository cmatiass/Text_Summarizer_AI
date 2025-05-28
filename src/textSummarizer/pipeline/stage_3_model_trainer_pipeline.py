
"""
Stage 3: Model Trainer Pipeline
This module implements the third stage of the ML pipeline for text summarization.
It handles the training of the text summarization model using the preprocessed data.
"""

# Import necessary modules for configuration management, model trainer component, and logging
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_trainer import ModelTrainer
from src.textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    """
    Pipeline class for handling model training stage of the training process.
    
    This class orchestrates the model training process which includes:
    1. Loading configuration settings for model training
    2. Setting up the training parameters and hyperparameters
    3. Training the text summarization model on the processed data
    4. Saving the trained model for evaluation and inference
    """
    
    def __init__(self):
        """
        Initialize the ModelTrainerTrainingPipeline.
        Currently, no initialization parameters are required.
        """
        pass
    
    def initiate_model_trainer(self):
        """
        Execute the complete model training process.
        
        This method performs the following steps:
        1. Creates a configuration manager to load settings
        2. Retrieves model trainer specific configuration (hyperparameters, paths, etc.)
        3. Initializes the model trainer component
        4. Starts the training process using the processed data
        
        The training process typically includes:
        - Loading the pre-trained base model (e.g., PEGASUS)
        - Fine-tuning on the domain-specific summarization dataset
        - Applying training hyperparameters (learning rate, batch size, epochs, etc.)
        - Saving checkpoints and the final trained model
        """
        # Initialize configuration manager to access project settings
        config = ConfigurationManager()
        
        # Get specific configuration parameters for model training
        model_trainer_config = config.get_model_trainer_config()
        
        # Create model trainer component with the loaded configuration
        model_trainer = ModelTrainer(config=model_trainer_config)  # Fixed variable name
        
        # Execute the model training process
        model_trainer.train()