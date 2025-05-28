
"""
Stage 2: Data Transformation Pipeline
This module implements the second stage of the ML pipeline for text summarization.
It handles preprocessing and transforming the raw data into a format suitable for model training.
"""

# Import necessary modules for configuration management, data transformation component, and logging
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.data_transformation import DataTransformation
from src.textSummarizer.logging import logger


class DataTransformationTrainingPipeline:
    """
    Pipeline class for handling data transformation stage of the training process.
    
    This class orchestrates the data transformation process which includes:
    1. Loading configuration settings for data transformation
    2. Converting raw data into tokenized format suitable for training
    3. Preparing data in the required format for the text summarization model
    """
    
    def __init__(self):
        """
        Initialize the DataTransformationTrainingPipeline.
        Currently, no initialization parameters are required.
        """
        pass

    def initiate_data_transformation(self):
        """
        Execute the complete data transformation process.
        
        This method performs the following steps:
        1. Creates a configuration manager to load settings
        2. Retrieves data transformation specific configuration
        3. Initializes the data transformation component
        4. Converts the raw data into processed format for model training
        
        The transformation typically includes:
        - Tokenization of text data
        - Creating input-output pairs for summarization
        - Formatting data according to model requirements
        - Saving processed data for training stage
        """
        # Initialize configuration manager to access project settings
        config = ConfigurationManager()
        
        # Get specific configuration parameters for data transformation
        data_transformation_config = config.get_data_transformation_config()
        
        # Create data transformation component with the loaded configuration
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # Execute the data conversion/transformation process
        data_transformation.convert()
        