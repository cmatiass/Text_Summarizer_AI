
"""
Entity Configuration Classes for Text Summarizer AI

This module defines data classes that represent configuration entities for different stages
of the text summarization ML pipeline. These classes serve as structured containers for
configuration parameters, ensuring type safety and clear organization of settings.

Each configuration class corresponds to a specific pipeline stage:
- DataIngestionConfig: Settings for downloading and extracting datasets
- DataTransformationConfig: Parameters for data preprocessing and tokenization
- ModelTrainerConfig: Training hyperparameters and model configuration
- ModelEvaluationConfig: Evaluation settings and output paths

Using dataclasses provides automatic initialization, string representation, and
type hints for better code maintainability and debugging.
"""

# Import dataclass decorator for automatic class generation and Path for file system operations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    """
    Configuration class for the data ingestion stage.
    
    This class holds all necessary parameters for downloading and extracting
    the training dataset from external sources.
    
    Attributes:
        root_dir (Path): Root directory where all data ingestion files will be stored
        source_URL (Path): URL or path to the source dataset (typically a downloadable zip file)
        local_data_file (Path): Local path where the downloaded file will be saved
        unzip_dir (Path): Directory where the downloaded zip file will be extracted
    """
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path


@dataclass
class DataTransformationConfig:
    """
    Configuration class for the data transformation stage.
    
    This class contains parameters needed for preprocessing raw data into
    a format suitable for model training, including tokenization settings.
    
    Attributes:
        root_dir (Path): Root directory for data transformation outputs
        data_path (Path): Path to the raw data that needs to be transformed
        tokenizer_name (Path): Name or path of the tokenizer to use for text processing
                              (e.g., "google/pegasus-cnn_dailymail")
    """
    root_dir: Path
    data_path: Path
    tokenizer_name: Path


@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the model training stage.
    
    This class encapsulates all hyperparameters and settings required for
    training the text summarization model. It includes training dynamics,
    optimization parameters, and checkpoint management settings.
    
    Attributes:
        root_dir (Path): Root directory for training outputs and checkpoints
        data_path (Path): Path to the preprocessed training data
        model_ckpt (Path): Path or name of the pre-trained model checkpoint to fine-tune
        num_train_epochs (int): Number of complete passes through the training dataset
        warmup_steps (int): Number of steps for learning rate warmup at training start
        per_device_train_batch_size (int): Batch size per GPU/CPU device during training
        weight_decay (float): L2 regularization coefficient to prevent overfitting
        logging_steps (int): Frequency of logging training metrics (every N steps)
        evaluation_strategy (str): When to run evaluation ("steps", "epoch", or "no")
        eval_steps (int): Number of steps between evaluations (if evaluation_strategy="steps")
        save_steps (float): Frequency of saving model checkpoints during training
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating
                                          (useful for simulating larger batch sizes)
    """
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Configuration class for the model evaluation stage.
    
    This class is marked as frozen (immutable) to ensure evaluation settings
    remain constant throughout the evaluation process. It contains paths to
    the trained model, test data, and output locations for evaluation results.
    
    Attributes:
        root_dir (Path): Root directory for evaluation outputs and results
        data_path (Path): Path to the test/validation dataset for evaluation
        model_path (Path): Path to the trained model that will be evaluated
        tokenizer_path (Path): Path to the tokenizer associated with the trained model
        metric_file_name (Path): Filename where evaluation metrics will be saved
                                (typically contains ROUGE scores, BLEU scores, etc.)
    
    Note: This class is frozen to prevent accidental modification of evaluation
          parameters, ensuring reproducible evaluation results.
    """
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path