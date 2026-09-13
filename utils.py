# utils.py
import logging
import yaml

def load_configuration():
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def log_message(message):
    logger = logging.getLogger(__name__)
    logger.info(message)

def handle_error(error):
    logger = logging.getLogger(__name__)
    logger.error(f"An error occurred: {error}")
    # Additional error handling logic can be added here
