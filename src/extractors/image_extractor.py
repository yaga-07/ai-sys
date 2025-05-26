from typing import List, Dict, Any
from src.core.extractor import BaseExtractor
from src.core.llm_engine import BaseLLMEngine

import logging
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ImageExtractorAI(BaseExtractor):
    def __init__(self, llm_engine: 'BaseLLMEngine'):
        self.llm_engine = llm_engine

    def extract(self, image_bytes: bytes) -> str:
        """
        Extract text from an image file using an AI model.
        :param image_bytes: The bytes of the image file.
        :return: Extracted text as a string.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI model that extracts text from images."
                },
                {
                    "role": "user",
                    "content": f"Extract text from the following image as it is"
                }
            ]

            response = self.llm_engine.generate_from_image(image_bytes=image_bytes, messages=messages)
            logger.info("Text extraction from image completed successfully.")
        except Exception as e:
            logger.error(f"Failed to extract text from image: {e}")
            raise RuntimeError(f"Failed to extract text from image: {e}")

        return response