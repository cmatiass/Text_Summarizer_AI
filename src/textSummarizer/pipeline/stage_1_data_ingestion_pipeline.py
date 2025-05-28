
"""
Stage 1: Data Ingestion Pipeline
This module implements the first stage of the ML pipeline for text summarization.
It handles downloading and extracting the dataset required for training.
"""

# Import necessary modules for configuration management, data ingestion component, and logging
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.data_ingestion import DataIngestion
from src.textSummarizer.logging import logger


class DataIngestionTrainingPipeline:
    """
    Pipeline class for handling data ingestion stage of the training process.
    
    This class orchestrates the data ingestion process which includes:
    1. Loading configuration settings for data ingestion
    2. Downloading the dataset from the specified source
    3. Extracting the downloaded zip file to make data accessible
    """
    
    def __init__(self):
        """
        Initialize the DataIngestionTrainingPipeline.
        Currently, no initialization parameters are required.
        """
        pass

    def initiate_data_ingestion(self):
        """
        Execute the complete data ingestion process.
        
        This method performs the following steps:
        1. Creates a configuration manager to load settings
        2. Retrieves data ingestion specific configuration
        3. Initializes the data ingestion component
        4. Downloads the dataset file from the configured source
        5. Extracts the downloaded zip file to the target directory
        
        The process ensures that the raw data is available for subsequent
        pipeline stages (data transformation, model training, etc.)
        """
        # Initialize configuration manager to access project settings
        config = ConfigurationManager()
        
        # Get specific configuration parameters for data ingestion
        data_ingestion_config = config.get_data_ingestion_config()
        
        # Create data ingestion component with the loaded configuration
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Download the dataset file from the configured URL
        data_ingestion.downlaod_file()  # Note: There's a typo in the original method name
        
        # Extract the downloaded zip file to make the data accessible
        data_ingestion.extract_zip_file()