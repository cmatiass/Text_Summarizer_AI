
"""
Data Transformation Component for Text Summarizer AI

This module implements the data transformation functionality for the text summarization pipeline.
It handles preprocessing raw text data into tokenized format suitable for model training.

Key responsibilities:
- Load and tokenize text data using pre-trained tokenizers
- Convert dialogue-summary pairs into model-ready input-output format
- Apply proper sequence length limits and truncation strategies
- Transform datasets into the format expected by transformer models
- Save processed datasets for efficient loading during training

This component bridges the gap between raw text data and model training,
ensuring that data is properly formatted for the PEGASUS or similar transformer models.
"""

# Import required modules for file operations and logging
import os
from src.textSummarizer.logging import logger

# Import transformer and dataset libraries for text processing
from transformers import AutoTokenizer
from datasets import load_from_disk

# Import project-specific configuration
from src.textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    """
    Data transformation component responsible for preprocessing text data for model training.
    
    This class handles the second stage of the ML pipeline by:
    1. Loading pre-trained tokenizers for text processing
    2. Converting raw dialogue-summary pairs into tokenized format
    3. Applying appropriate sequence length limits and attention masks
    4. Preparing data in the exact format required by transformer models
    5. Saving processed datasets for efficient training data loading
    
    The component is specifically designed for dialogue summarization tasks
    but can be adapted for other text-to-text generation tasks.
    """
    
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize the DataTransformation component with configuration and tokenizer.
        
        Args:
            config (DataTransformationConfig): Configuration object containing:
                - data_path: Path to the raw dataset to be processed
                - tokenizer_name: Name/path of the tokenizer to use
                - root_dir: Directory to save processed datasets
                
        The initialization loads the specified tokenizer which will be used
        for all text processing operations in this component.
        """
        # Store configuration for access to paths and settings
        self.config = config
        
        # Load the pre-trained tokenizer for text processing
        # This tokenizer should match the model that will be used for training
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        """
        Convert raw dialogue-summary pairs into tokenized features for model training.
        
        This method processes batches of examples by:
        1. Tokenizing input dialogues with appropriate length limits
        2. Tokenizing target summaries using special target tokenizer mode
        3. Creating attention masks for proper sequence handling
        4. Formatting everything into the expected model input structure
        
        Args:
            example_batch: Batch of examples containing 'dialogue' and 'summary' keys
            
        Returns:
            dict: Dictionary containing tokenized features:
                - input_ids: Tokenized dialogue sequences
                - attention_mask: Masks indicating real vs padded tokens
                - labels: Tokenized summary sequences (targets for training)
                
        The method applies truncation to handle sequences longer than model limits
        and uses the tokenizer's special target mode for proper summary encoding.
        """
        # Tokenize input dialogues with maximum length limit and truncation
        # Max length of 1024 tokens is typical for PEGASUS and similar models
        input_encodings = self.tokenizer(
            example_batch['dialogue'], 
            max_length=1024,        # Maximum input sequence length
            truncation=True         # Truncate sequences that exceed max_length
        )

        # Tokenize target summaries using special target tokenizer mode
        # This ensures proper handling of decoder input formatting
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'], 
                max_length=128,     # Maximum summary length (shorter than input)
                truncation=True     # Truncate summaries that exceed max_length
            )

        # Return the processed features in the format expected by the model
        return {
            'input_ids': input_encodings['input_ids'],           # Tokenized dialogue
            'attention_mask': input_encodings['attention_mask'], # Attention masks
            'labels': target_encodings['input_ids']              # Tokenized summaries (targets)
        }
    
    def convert(self):
        """
        Execute the complete data transformation process.
        
        This method orchestrates the entire transformation pipeline:
        1. Loads the raw dataset from disk
        2. Applies tokenization transformation to all examples
        3. Saves the processed dataset for training use
        
        The processed dataset will contain tokenized input-output pairs
        ready for model training, with proper attention masks and labels.
        
        The transformation is applied in batches for efficiency, and the
        resulting dataset is saved in a format optimized for fast loading
        during the training process.
        """
        # Load the raw dataset from the configured path
        # This dataset should contain 'dialogue' and 'summary' columns
        dataset_samsum = load_from_disk(self.config.data_path)
        
        # Apply tokenization transformation to the entire dataset
        # batched=True enables efficient batch processing for better performance
        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features, 
            batched=True    # Process multiple examples at once for efficiency
        )
        
        # Save the processed dataset to disk for use in training
        # The processed dataset contains tokenized features ready for model input
        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )


    