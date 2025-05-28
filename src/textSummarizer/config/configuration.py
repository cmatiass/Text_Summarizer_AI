
"""
Configuration Manager for Text Summarizer AI

This module provides a centralized configuration management system for the entire
text summarization ML pipeline. It handles loading, parsing, and distributing
configuration parameters from YAML files to different pipeline stages.

The ConfigurationManager class serves as a factory for creating stage-specific
configuration objects, ensuring consistent parameter management across:
- Data ingestion (downloading and extracting datasets)
- Data transformation (preprocessing and tokenization)
- Model training (hyperparameters and training settings)
- Model evaluation (evaluation metrics and output paths)

Key features:
- YAML-based configuration loading
- Automatic directory creation for artifacts
- Type-safe configuration objects
- Separation of general config and training parameters
"""

# Import constants that define file paths for configuration files
from src.textSummarizer.constants import *

# Import utility functions for YAML parsing and directory management
from src.textSummarizer.utils.common import read_yaml, create_directories

# Import configuration dataclasses for type-safe parameter management
from src.textSummarizer.entity import DataIngestionConfig, ModelEvaluationConfig, DataTransformationConfig, ModelTrainerConfig


class ConfigurationManager:
    """
    Central configuration manager for the text summarization pipeline.
    
    This class provides a unified interface for loading and managing configuration
    parameters from YAML files. It handles the creation of stage-specific configuration
    objects and ensures proper directory structure setup.
    
    The manager loads two types of configuration files:
    1. Main config file (config.yaml): Contains paths, URLs, and general settings
    2. Parameters file (params.yaml): Contains hyperparameters for model training
    
    Attributes:
        config: Parsed configuration from the main config YAML file
        params: Parsed parameters from the parameters YAML file
    """
    
    def __init__(self,
                 config_path=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        """
        Initialize the ConfigurationManager with configuration file paths.
        
        Args:
            config_path: Path to the main configuration YAML file (default from constants)
            params_filepath: Path to the parameters YAML file (default from constants)
            
        The initialization process:
        1. Loads and parses both YAML configuration files
        2. Creates the main artifacts directory structure
        3. Prepares the manager for serving stage-specific configurations
        """
        # Load the main configuration file containing paths, URLs, and general settings
        self.config = read_yaml(config_path)
        
        # Load the parameters file containing training hyperparameters
        self.params = read_yaml(params_filepath)

        # Create the root artifacts directory where all pipeline outputs will be stored
        # This ensures the basic directory structure exists before any pipeline stage runs
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Create and return configuration object for the data ingestion stage.
        
        Returns:
            DataIngestionConfig: Configuration object containing all parameters
                                needed for downloading and extracting the dataset
                                
        This method:
        1. Extracts data ingestion settings from the main config
        2. Creates necessary directories for data storage
        3. Returns a typed configuration object with all required paths and URLs
        """
        # Extract data ingestion specific configuration section
        config = self.config.data_ingestion
        
        # Ensure the data ingestion root directory exists
        create_directories([config.root_dir])

        # Create and populate the data ingestion configuration object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,           # Directory for all ingestion artifacts
            source_URL=config.source_URL,       # URL to download the dataset from
            local_data_file=config.local_data_file,  # Local path to save downloaded file
            unzip_dir=config.unzip_dir          # Directory to extract the dataset to
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Create and return configuration object for the data transformation stage.
        
        Returns:
            DataTransformationConfig: Configuration object containing all parameters
                                    needed for preprocessing and tokenizing the data
                                    
        This method:
        1. Extracts data transformation settings from the main config
        2. Creates necessary directories for transformation outputs
        3. Returns a typed configuration object with data paths and tokenizer settings
        """
        # Extract data transformation specific configuration section
        config = self.config.data_transformation

        # Ensure the data transformation root directory exists
        create_directories([config.root_dir])

        # Create and populate the data transformation configuration object
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,           # Directory for transformation artifacts
            data_path=config.data_path,         # Path to the raw data to be processed
            tokenizer_name=config.tokenizer_name  # Name/path of the tokenizer to use
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Create and return configuration object for the model training stage.
        
        Returns:
            ModelTrainerConfig: Configuration object containing all parameters
                              needed for training the text summarization model
                              
        This method:
        1. Extracts model training settings from the main config
        2. Extracts training hyperparameters from the params file
        3. Creates necessary directories for training outputs
        4. Returns a comprehensive configuration object with all training parameters
        
        The configuration combines:
        - Basic settings (paths, model checkpoint) from config.yaml
        - Training hyperparameters from params.yaml
        """
        # Extract model trainer specific configuration section
        config = self.config.model_trainer
        
        # Extract training hyperparameters from the parameters file
        params = self.params.TrainingArguments

        # Ensure the model trainer root directory exists
        create_directories([config.root_dir])

        # Create and populate the model trainer configuration object
        # Combines basic config with detailed training hyperparameters
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,                               # Directory for training artifacts
            data_path=config.data_path,                             # Path to preprocessed training data
            model_ckpt=config.model_ckpt,                          # Pre-trained model checkpoint to fine-tune
            num_train_epochs=params.num_train_epochs,              # Number of training epochs
            warmup_steps=params.warmup_steps,                      # Learning rate warmup steps
            per_device_train_batch_size=params.per_device_train_batch_size,  # Batch size per device
            weight_decay=params.weight_decay,                      # L2 regularization coefficient
            logging_steps=params.logging_steps,                    # Frequency of logging metrics
            evaluation_strategy=params.evaluation_strategy,        # When to evaluate ("steps" or "epoch")
            eval_steps=params.evaluation_strategy,                 # Steps between evaluations
            save_steps=params.save_steps,                         # Frequency of saving checkpoints
            gradient_accumulation_steps=params.gradient_accumulation_steps  # Gradient accumulation steps
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Create and return configuration object for the model evaluation stage.
        
        Returns:
            ModelEvaluationConfig: Configuration object containing all parameters
                                 needed for evaluating the trained model
                                 
        This method:
        1. Extracts model evaluation settings from the main config
        2. Creates necessary directories for evaluation outputs
        3. Returns a typed configuration object with model paths and evaluation settings
        
        The configuration includes paths to:
        - Trained model and tokenizer
        - Test/validation data
        - Output file for evaluation metrics
        """
        # Extract model evaluation specific configuration section
        config = self.config.model_evaluation

        # Ensure the model evaluation root directory exists
        create_directories([config.root_dir])

        # Create and populate the model evaluation configuration object
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,                    # Directory for evaluation artifacts
            data_path=config.data_path,                  # Path to test/validation data
            model_path=config.model_path,                # Path to the trained model
            tokenizer_path=config.tokenizer_path,        # Path to the model's tokenizer
            metric_file_name=config.metric_file_name     # File to save evaluation metrics
        )

        return model_evaluation_config


    

    
