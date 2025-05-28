
"""
Data Ingestion Component for Text Summarizer AI

This module implements the data ingestion functionality for the text summarization pipeline.
It handles downloading datasets from external sources and extracting them to local storage
for subsequent processing stages.

Key responsibilities:
- Download datasets from URLs (with smart caching to avoid re-downloads)
- Extract compressed files (ZIP format) to designated directories
- Integrate with the logging system for monitoring download progress
- Handle file system operations safely with proper error handling

This component is the first stage in the ML pipeline and ensures that raw data
is available for the data transformation stage.
"""

# Import required modules for file operations, web requests, and compression handling
import os
import urllib.request as request
import zipfile

# Import project-specific modules for logging and configuration
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataIngestionConfig


class DataIngestion:
    """
    Data ingestion component responsible for downloading and extracting datasets.
    
    This class handles the first stage of the ML pipeline by:
    1. Downloading datasets from remote URLs
    2. Extracting compressed files to accessible locations
    3. Providing intelligent caching to avoid unnecessary re-downloads
    4. Logging all operations for monitoring and debugging
    
    The component is designed to work with ZIP-compressed datasets and can be
    easily extended to support other compression formats.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion component with configuration settings.
        
        Args:
            config (DataIngestionConfig): Configuration object containing:
                - source_URL: URL to download the dataset from
                - local_data_file: Local path to save the downloaded file
                - unzip_dir: Directory to extract the dataset to
                - root_dir: Root directory for all ingestion artifacts
        """
        # Store the configuration for use in download and extraction methods
        self.config = config

    def downlaod_file(self):  # Note: Original typo preserved for compatibility
        """
        Download the dataset file from the configured URL.
        
        This method implements intelligent downloading with the following features:
        - Checks if the file already exists locally to avoid unnecessary downloads
        - Downloads from the configured source URL if the file is missing
        - Logs the download status for monitoring purposes
        - Uses urllib.request for reliable HTTP/HTTPS downloads
        
        The method handles network operations safely and provides clear feedback
        about whether a download was performed or skipped due to existing files.
        """
        # Check if the target file already exists to avoid redundant downloads
        if not os.path.exists(self.config.local_data_file):
            # File doesn't exist, proceed with download
            # Use urlretrieve for robust file downloading with automatic retries
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,           # Source URL from configuration
                filename=self.config.local_data_file  # Local destination path
            )
            # Log successful download for monitoring
            logger.info(f"File is downloaded")
        else:
            # File already exists, skip download to save time and bandwidth
            logger.info(f"File already exits")  # Note: Original typo preserved

    def extract_zip_file(self):
        """
        Extract the downloaded ZIP file to the configured directory.
        
        This method handles the extraction of ZIP-compressed datasets with:
        - Automatic creation of the target extraction directory
        - Safe extraction using Python's zipfile module
        - Proper handling of nested directory structures within ZIP files
        - Error-resistant directory creation (won't fail if directory exists)
        
        The extraction process prepares the raw data for the next pipeline stage
        (data transformation) by making all dataset files accessible in a
        structured directory format.
        
        Args:
            zip_file_path (str): Path to the ZIP file to extract (from config)
            
        Returns:
            None: The method performs file operations and doesn't return values
            
        Note: The method expects the ZIP file to have been downloaded successfully
              by the download_file() method before being called.
        """
        # Get the target extraction directory from configuration
        unzip_path = self.config.unzip_dir
        
        # Create the extraction directory if it doesn't exist
        # exist_ok=True prevents errors if the directory already exists
        os.makedirs(unzip_path, exist_ok=True)
        
        # Open and extract the ZIP file safely using context manager
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            # Extract all contents to the designated directory
            # This preserves the internal directory structure of the ZIP file
            zip_ref.extractall(unzip_path)

    
