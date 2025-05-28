
"""
Stage 4: Model Evaluation Pipeline
This module implements the fourth stage of the ML pipeline for text summarization.
It handles the evaluation of the trained model using various metrics and test data.
"""

# Import necessary modules for configuration management, model evaluation component, and logging
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_evaluation import ModelEvaluation
from src.textSummarizer.logging import logger


class ModelEvaluationTrainingPipeline:
    """
    Pipeline class for handling model evaluation stage of the training process.
    
    This class orchestrates the model evaluation process which includes:
    1. Loading configuration settings for model evaluation
    2. Setting up evaluation metrics and test data
    3. Evaluating the trained model's performance
    4. Generating evaluation reports and metrics
    """
    
    def __init__(self):
        """
        Initialize the ModelEvaluationTrainingPipeline.
        Currently, no initialization parameters are required.
        """
        pass

    def initiate_model_evaluation(self):
        """
        Execute the complete model evaluation process.
        
        This method performs the following steps:
        1. Creates a configuration manager to load settings
        2. Retrieves model evaluation specific configuration
        3. Initializes the model evaluation component
        4. Runs evaluation on the trained model
        
        The evaluation process typically includes:
        - Loading the trained model and test dataset
        - Generating summaries on test data
        - Calculating evaluation metrics (ROUGE scores, BLEU, etc.)
        - Saving evaluation results and metrics to files
        - Providing insights into model performance
        """
        # Initialize configuration manager to access project settings
        config = ConfigurationManager()
        
        # Get specific configuration parameters for model evaluation
        model_evaluation_config = config.get_model_evaluation_config()
        
        # Create model evaluation component with the loaded configuration
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
        # Execute the model evaluation process
        model_evaluation.evaluate()