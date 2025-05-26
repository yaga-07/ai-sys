from typing import List, Dict, Any
from src.core.extractor import BaseExtractor
from src.core.llm_engine import BaseLLMEngine

import logging
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PDFExtractor(BaseExtractor):
    def extract(self, file_path: str) -> str:
        # Implement PDF extraction logic here
        return "PDF text"

class PDFExtractionAI(BaseExtractor):
    def __init__(self, llm_engine: 'BaseLLMEngine'):
        self.llm_engine = llm_engine
        
    def extract(self, pdf_bytes: bytes) -> str:
        """
        Extract text from a PDF file using an AI model.
        :param pdf_bytes: The bytes of the PDF file.
        :return: Extracted text as a string.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI model that extracts text from PDF files."
                },
                {
                    "role": "user",
                    "content": f"Extract text from the following PDF as it is"
                }
            ]

            response = self.llm_engine.generate_from_pdf(pdf_bytes=pdf_bytes, messages=messages)
            logger.info("Text extraction from PDF completed successfully.")
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

        return response