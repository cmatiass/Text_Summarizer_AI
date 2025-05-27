from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline
import os


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self,text):
        # Check if local tokenizer exists, otherwise use the original pre-trained model
        if os.path.exists(self.config.tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, local_files_only=True)
        else:
            # Fallback to original pre-trained tokenizer
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}        # Check if local model exists, otherwise use the original pre-trained model  
        if os.path.exists(self.config.model_path):
            pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)
        else:
            # Fallback to original pre-trained model
            pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail", tokenizer=tokenizer)
        
        print("Dialogue:")
        print(text)
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        
        # Clean the output by removing <n> markers and extra spaces
        cleaned_output = output.replace("<n>", " ").strip()
        # Remove multiple spaces with single space
        while "  " in cleaned_output:
            cleaned_output = cleaned_output.replace("  ", " ")
        
        print("\nModel Summary:")
        print(cleaned_output)

        return cleaned_output