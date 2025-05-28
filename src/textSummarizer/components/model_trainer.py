
"""
Model Trainer Component for Text Summarizer AI

This module implements the model training functionality for the text summarization pipeline.
It handles fine-tuning pre-trained transformer models (like PEGASUS) on domain-specific data.

Key responsibilities:
- Load and configure pre-trained sequence-to-sequence models
- Set up training arguments and optimization parameters
- Handle GPU/CPU device selection and model placement
- Configure data collation for efficient batch processing
- Execute the training process with proper monitoring
- Save trained models and tokenizers for later use

This component represents the core of the ML pipeline where the actual learning happens,
transforming a general-purpose model into a domain-specific text summarizer.
"""

# Import transformer library components for model training
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq

# Import PyTorch for device management and tensor operations
import torch

# Import dataset utilities and file system operations
from datasets import load_from_disk
import os

# Import project-specific configuration
from src.textSummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    """
    Model trainer component responsible for fine-tuning transformer models for text summarization.
    
    This class handles the third stage of the ML pipeline by:
    1. Loading pre-trained models and tokenizers
    2. Setting up training infrastructure (device selection, data collation)
    3. Configuring training parameters and optimization strategies
    4. Executing the fine-tuning process on processed data
    5. Saving the trained model and tokenizer for evaluation and inference
    
    The component is designed to work with sequence-to-sequence transformer models
    like PEGASUS, T5, or BART, and handles both GPU and CPU training scenarios.
    """
    
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize the ModelTrainer component with training configuration.
        
        Args:
            config (ModelTrainerConfig): Configuration object containing:
                - model_ckpt: Pre-trained model checkpoint name/path
                - data_path: Path to the processed training dataset
                - root_dir: Directory to save training outputs
                - Training hyperparameters (epochs, batch size, learning rate, etc.)
        """
        # Store configuration for access to training parameters and paths
        self.config = config

    def train(self):
        """
        Execute the complete model training process.
        
        This method orchestrates the entire training pipeline:
        1. Sets up the computing device (GPU if available, otherwise CPU)
        2. Loads the pre-trained model and tokenizer
        3. Configures data collation for efficient batch processing
        4. Loads the processed training dataset
        5. Sets up training arguments with hyperparameters
        6. Initializes the Trainer with all components
        7. Executes the training process
        8. Saves the fine-tuned model and tokenizer
        
        The training process uses the Hugging Face Trainer API which handles:
        - Automatic mixed precision training
        - Gradient accumulation and clipping
        - Learning rate scheduling
        - Evaluation during training
        - Checkpoint saving and recovery
        """
        # Determine the best available device for training
        # Use GPU if available for faster training, otherwise fall back to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the pre-trained tokenizer for text processing
        # This tokenizer matches the model architecture and vocabulary
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        
        # Load the pre-trained sequence-to-sequence model and move it to the selected device
        # This model will be fine-tuned on the domain-specific summarization data
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        
        # Set up data collator for efficient batch processing
        # This handles padding, attention masks, and proper formatting for seq2seq training
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # Load the processed training dataset from disk
        # This dataset should contain tokenized input-output pairs ready for training
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Configure training arguments with hyperparameters and optimization settings
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,                    # Directory for training outputs
            num_train_epochs=1,                                 # Number of training epochs
            warmup_steps=500,                                   # Learning rate warmup steps
            per_device_train_batch_size=1,                     # Training batch size per device
            per_device_eval_batch_size=1,                      # Evaluation batch size per device
            weight_decay=0.01,                                  # L2 regularization coefficient
            logging_steps=10,                                   # Frequency of logging metrics
            eval_strategy='steps',                              # Evaluate every N steps
            eval_steps=500,                                     # Steps between evaluations
            save_steps=1e6,                                     # Steps between checkpoint saves
            gradient_accumulation_steps=16                      # Gradient accumulation for larger effective batch size
        )
        
        # Initialize the Trainer with model, arguments, and data components
        # The Trainer handles the training loop, optimization, and monitoring
        trainer = Trainer(
            model=model_pegasus,                               # Model to train
            args=trainer_args,                                 # Training configuration
            tokenizer=tokenizer,                               # Tokenizer for text processing
            data_collator=seq2seq_data_collator,              # Data collation strategy
            train_dataset=dataset_samsum_pt["test"],           # Training data split
            eval_dataset=dataset_samsum_pt["validation"]       # Validation data for monitoring
        )
        
        # Execute the training process
        # This runs the complete training loop with automatic optimization,
        # evaluation, and checkpoint management
        trainer.train()

        # Save the fine-tuned model to disk for later use
        # This creates a complete model checkpoint that can be loaded for inference
        model_pegasus.save_pretrained(
            os.path.join(self.config.root_dir, "pegasus-samsum-model")
        )
        
        # Save the tokenizer alongside the model
        # This ensures consistency between training and inference tokenization
        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )


