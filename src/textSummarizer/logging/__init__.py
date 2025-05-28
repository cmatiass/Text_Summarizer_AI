
"""
Logging Configuration Module for Text Summarizer AI

This module sets up a centralized logging system for the entire text summarization project.
It configures both file and console logging to track application behavior, errors, and progress
throughout the ML pipeline stages (data ingestion, transformation, training, evaluation, and prediction).

The logging system provides:
- Structured log format with timestamps, log levels, module names, and messages
- Dual output: both to log files and console for comprehensive monitoring
- Centralized logger instance that can be imported across all project modules
"""

# Import required modules for file operations, system output, and logging functionality
import os
import sys
import logging

# Configuration for log directory and file storage
log_dir = "logs"  # Directory name where all log files will be stored

# Define the log message format with useful debugging information
# Format includes: timestamp, log level (INFO/ERROR/DEBUG), module name, and the actual message
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Construct the full path for the continuous log file
# This file will contain all logging output from the application
log_filepath = os.path.join(log_dir, "continuos_logs.log")  # Note: Original typo preserved

# Create the logs directory if it doesn't exist
# exist_ok=True prevents errors if the directory already exists
os.makedirs(log_dir, exist_ok=True)

# Configure the global logging system with comprehensive settings
logging.basicConfig(
    level=logging.INFO,          # Set minimum log level to INFO (excludes DEBUG messages)
    format=logging_str,          # Apply the custom format defined above
    
    # Set up dual logging handlers for maximum visibility
    handlers=[
        # File handler: writes all logs to the specified file for persistent storage
        logging.FileHandler(log_filepath),
        
        # Stream handler: displays logs in real-time on the console/terminal
        # This allows developers to see logs immediately during development and debugging
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a named logger instance specifically for the text summarizer project
# This logger can be imported and used throughout all project modules
# Using a named logger helps identify which application generated the logs
logger = logging.getLogger("summarizerlogger")
