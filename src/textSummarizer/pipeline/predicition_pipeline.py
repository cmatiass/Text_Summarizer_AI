
"""
Prediction Pipeline
This module implements the inference pipeline for text summarization.
It handles loading the trained model and generating summaries for new text inputs.
"""

# Import necessary modules for configuration, transformers library, and OS operations
from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline
import os


class PredictionPipeline:
    """
    Pipeline class for handling text summarization inference/prediction.
    
    This class provides functionality to:
    1. Load trained model and tokenizer (with fallback to pre-trained models)
    2. Generate summaries for input text
    3. Clean and format the output summaries
    4. Handle both local fine-tuned models and original pre-trained models
    """
    
    def __init__(self):
        """
        Initialize the PredictionPipeline.
        
        Sets up the configuration manager and loads model evaluation config
        which contains paths to the trained model and tokenizer.
        """
        # Load configuration settings for model paths and parameters
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        """
        Generate a summary for the given input text.
        
        Args:
            text (str): The input text to be summarized
            
        Returns:
            str: The generated summary text, cleaned and formatted
            
        This method performs the following steps:
        1. Loads the appropriate tokenizer (local or pre-trained)
        2. Sets up generation parameters for summary quality
        3. Loads the appropriate model (local or pre-trained)
        4. Generates the summary using the transformers pipeline
        5. Cleans and formats the output text
        6. Returns the final summary
        """
        # Load tokenizer: prioritize local fine-tuned version, fallback to pre-trained
        if os.path.exists(self.config.tokenizer_path):
            # Use locally fine-tuned tokenizer if available
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, local_files_only=True)
        else:
            # Fallback to original pre-trained tokenizer from HuggingFace
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        
        # Set generation parameters for summary quality and length control
        gen_kwargs = {
            "length_penalty": 0.8,  # Penalty for length to encourage concise summaries
            "num_beams": 8,         # Number of beams for beam search (higher = better quality)
            "max_length": 128       # Maximum length of generated summary
        }
        
        # Load model: prioritize local fine-tuned version, fallback to pre-trained
        if os.path.exists(self.config.model_path):
            # Use locally fine-tuned model if available
            pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)
        else:
            # Fallback to original pre-trained model from HuggingFace
            pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail", tokenizer=tokenizer)
        
        # Display the input text for reference
        print("Dialogue:")
        print(text)
        
        # Generate summary using the loaded model and parameters
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        
        # Clean the output by removing special tokens and formatting issues
        cleaned_output = output.replace("<n>", " ").strip()  # Remove <n> tokens and trim whitespace
        
        # Remove multiple consecutive spaces with single spaces for better readability
        while "  " in cleaned_output:
            cleaned_output = cleaned_output.replace("  ", " ")
        
        # Display the generated summary
        print("\nModel Summary:")
        print(cleaned_output)

        # Return the cleaned summary text
        return cleaned_output