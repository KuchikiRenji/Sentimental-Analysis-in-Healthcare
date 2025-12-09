"""
Utility functions for the Sentimental Analysis in Healthcare project.
"""
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_nltk_resource(resource_name: str) -> None:
    """
    Safely download an NLTK resource if it doesn't already exist.
    
    Args:
        resource_name: Name of the NLTK resource to download (e.g., 'punkt', 'stopwords', 'vader_lexicon')
    
    Raises:
        Exception: If the download fails after retrying
    """
    # Map resources to their typical paths in NLTK data
    resource_paths = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'vader_lexicon': 'sentiment/vader_lexicon/vader_lexicon.txt'
    }
    
    # Check if resource already exists
    resource_path = resource_paths.get(resource_name, resource_name)
    try:
        nltk.data.find(resource_path)
        logger.debug(f"NLTK resource '{resource_name}' already exists")
        return
    except LookupError:
        pass
    
    # Resource doesn't exist, try to download it
    try:
        logger.info(f"Downloading NLTK resource '{resource_name}'...")
        nltk.download(resource_name, quiet=True)
        logger.info(f"Successfully downloaded NLTK resource '{resource_name}'")
    except Exception as e:
        logger.error(f"Failed to download NLTK resource '{resource_name}': {str(e)}")
        raise


def ensure_nltk_resources(resources: list) -> None:
    """
    Ensure all required NLTK resources are downloaded.
    
    Args:
        resources: List of NLTK resource names to check/download
    
    Raises:
        Exception: If any resource download fails
    """
    for resource in resources:
        download_nltk_resource(resource)

