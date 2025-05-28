
"""
Model Evaluation Component for Text Summarizer AI

This module implements the model evaluation functionality for the text summarization pipeline.
It handles comprehensive assessment of trained models using industry-standard metrics.

Key responsibilities:
- Load trained models and tokenizers for evaluation
- Generate summaries on test datasets using various decoding strategies
- Calculate ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum) for quality assessment
- Handle batch processing for efficient evaluation on large datasets
- Export evaluation results to structured formats (CSV) for analysis
- Support GPU acceleration for faster summary generation

This component provides quantitative measures of model performance, enabling
comparison between different models and tracking improvement over training iterations.
"""

# Import transformer library components for model loading and inference
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Import dataset utilities for loading test data
from datasets import load_from_disk

# Import PyTorch for device management and tensor operations
import torch

# Import data processing and analysis libraries
import pandas as pd
from tqdm import tqdm

# Import evaluation metrics library
import evaluate

# Import project-specific configuration
from src.textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    """
    Model evaluation component responsible for assessing trained text summarization models.
    
    This class handles the fourth stage of the ML pipeline by:
    1. Loading trained models and their associated tokenizers
    2. Processing test datasets in efficient batches
    3. Generating summaries using optimized decoding strategies
    4. Calculating comprehensive ROUGE metrics for quality assessment
    5. Exporting evaluation results for analysis and reporting
    
    The component supports both GPU and CPU evaluation and handles large datasets
    through intelligent batch processing to manage memory usage effectively.
    """
    
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize the ModelEvaluation component with evaluation configuration.
        
        Args:
            config (ModelEvaluationConfig): Configuration object containing:
                - model_path: Path to the trained model to evaluate
                - tokenizer_path: Path to the model's tokenizer
                - data_path: Path to the test dataset
                - metric_file_name: Output file for evaluation results
                - root_dir: Directory for evaluation outputs
        """
        # Store configuration for access to model paths and evaluation settings
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """
        Split a large dataset into smaller, manageable batches for processing.
        
        This utility method enables efficient processing of large datasets by:
        - Reducing memory usage through smaller batch sizes
        - Enabling progress tracking during evaluation
        - Preventing out-of-memory errors on limited hardware
        - Allowing parallel processing of independent batches
        
        Args:
            list_of_elements: The complete dataset or list to be split
            batch_size (int): Size of each batch chunk
            
        Yields:
            list: Successive batch-sized chunks from the input list
            
        This generator function is memory-efficient as it yields one batch at a time
        rather than creating all batches in memory simultaneously.
        """
        # Iterate through the list in steps of batch_size
        for i in range(0, len(list_of_elements), batch_size):
            # Yield a slice of the list from current position to current + batch_size
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                               batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                               column_text="article", 
                               column_summary="highlights"):
        """
        Calculate evaluation metrics on a test dataset using batch processing.
        
        This method orchestrates the complete evaluation process:
        1. Splits the dataset into manageable batches
        2. Generates summaries for each batch using the trained model
        3. Compares generated summaries with reference summaries
        4. Accumulates metric scores across all batches
        5. Returns comprehensive evaluation results
        
        Args:
            dataset: Test dataset containing input texts and reference summaries
            metric: Evaluation metric object (e.g., ROUGE metric)
            model: Trained model for generating summaries
            tokenizer: Tokenizer associated with the model
            batch_size (int): Number of examples to process per batch
            device (str): Computing device ("cuda" or "cpu")
            column_text (str): Name of the input text column in dataset
            column_summary (str): Name of the reference summary column
            
        Returns:
            dict: Computed metric scores (ROUGE-1, ROUGE-2, etc.)
            
        The method uses advanced generation parameters like beam search and length penalty
        to produce high-quality summaries that are then evaluated against references.
        """
        # Split input texts and reference summaries into corresponding batches
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        # Process each batch pair with progress tracking
        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            # Tokenize the input batch with padding and truncation for consistent tensor shapes
            inputs = tokenizer(
                article_batch, 
                max_length=1024,           # Maximum input sequence length
                truncation=True,           # Truncate sequences exceeding max_length
                padding="max_length",      # Pad shorter sequences to max_length
                return_tensors="pt"        # Return PyTorch tensors
            )
            
            # Generate summaries using the trained model with optimized parameters
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),           # Input token IDs
                attention_mask=inputs["attention_mask"].to(device), # Attention masks
                length_penalty=0.8,        # Penalty for length to encourage conciseness
                num_beams=8,              # Number of beams for beam search (higher = better quality)
                max_length=128            # Maximum length of generated summaries
            )
            # Length penalty ensures that the model does not generate sequences that are too long
            
            # Decode the generated token sequences back to readable text
            # Remove special tokens and clean up formatting for evaluation
            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for s in summaries
            ]      
            
            # Clean the decoded summaries by removing any remaining special tokens
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            # Add the batch predictions and references to the metric for accumulation
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        # Compute and return the final ROUGE scores across all processed batches
        score = metric.compute()
        return score
    
    def evaluate(self):
        """
        Execute the complete model evaluation process.
        
        This method orchestrates the entire evaluation pipeline:
        1. Sets up the computing environment (GPU/CPU)
        2. Loads the trained model and tokenizer
        3. Loads the test dataset for evaluation
        4. Initializes ROUGE metrics for comprehensive assessment
        5. Calculates metrics on a subset of test data for efficiency
        6. Exports results to CSV format for analysis
        
        The evaluation focuses on ROUGE metrics which are standard for
        summarization tasks, measuring overlap between generated and reference summaries.
        Results are saved to a CSV file for easy analysis and comparison.
        """
        # Determine the best available device for evaluation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the trained tokenizer from the specified path
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        # Load the trained model and move it to the selected device
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        # Load the test dataset for evaluation
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Define the ROUGE metrics to calculate for comprehensive evaluation
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        # Initialize the ROUGE metric evaluator
        rouge_metric = evaluate.load('rouge')

        # Calculate ROUGE scores on a subset of test data (first 10 examples for efficiency)
        # This provides a representative sample while keeping evaluation time manageable
        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10],   # Test subset for evaluation
            rouge_metric,                      # ROUGE metric evaluator
            model_pegasus,                     # Trained model
            tokenizer,                         # Model tokenizer
            batch_size=2,                      # Small batch size for memory efficiency
            column_text='dialogue',            # Input column name in dataset
            column_summary='summary'           # Reference summary column name
        )

        # Extract ROUGE scores into a structured dictionary
        # Directly use the scores without accessing sub-metrics (fmeasure, precision, recall)
        rouge_dict = {rn: score[rn] for rn in rouge_names}

        # Create a pandas DataFrame for structured data representation
        # Index with model name for easy identification in comparative analysis
        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        
        # Export evaluation results to CSV file for analysis and reporting
        df.to_csv(self.config.metric_file_name, index=False)