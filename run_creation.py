"""
Main script: launches the creation of the dataset
"""

from synthid.document_creation import create_document_samples
from logger import LOGGER


def main():
    LOGGER.info('Starting procedural creation of personal documents dataset.')
    create_document_samples()


if __name__ == '__main__':
    main()
