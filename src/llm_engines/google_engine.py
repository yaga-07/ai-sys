import logging
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

import vertexai
from google import genai, generativeai
from google.cloud import aiplatform
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Part

from src.core.llm_engine import BaseLLMEngine

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class GoogleLLMEngine(BaseLLMEngine):
    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            self.model = model
            self.api_key = api_key
            self.client = None
            self.credentials = None
            self.client_type = None
            self.logger = logger  # Use the module logger

            project_id = os.environ.get("GOOGLE_PROJECT_ID")
            location = os.environ.get("GOOGLE_LOCATION")
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

            # If API key is provided, try to initialize genai client directly
            if self.api_key or os.environ.get("GOOGLE_API_KEY", ""):
                api_key = self.api_key or os.environ.get("GOOGLE_API_KEY", "")
                try:
                    self.client = genai.Client(api_key=api_key)
                    self.client_type = "genai"
                    self.logger.info("Successfully initialized direct genai client using API key.")
                except Exception as e:
                    self.logger.error(f"Failed to initialize genai client with API key: {e}")
                    self.client = None
                    self.client_type = None
            # Only try Vertex AI if all three env vars are present
            elif project_id and location and credentials_path:
                try:
                    self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
                    self.logger.info("Loaded service account credentials from file.")
                    aiplatform.init(project=project_id, location=location, credentials=self.credentials)
                    self.client = GenerativeModel(model_name=self.model)
                    self.client_type = "vertexai"
                    self.logger.info(f"Successfully initialized Gemini client via Vertex AI using model {self.model}")
                except Exception as e:
                    self.logger.error(f"Error initializing Gemini client: {e}")
                    self.logger.error("\nPlease verify:")
                    self.logger.error(f"1. Project ID is correct: {project_id}")
                    self.logger.error("2. You have necessary IAM permissions")
                    self.logger.error("3. Vertex AI API is enabled in your project")
                    self.logger.error("4. GOOGLE_APPLICATION_CREDENTIALS is properly set")
                    self.client = None
                    self.client_type = None
            else:
                self.logger.warning("Insufficient credentials for Vertex AI (need GOOGLE_PROJECT_ID, GOOGLE_LOCATION, and GOOGLE_APPLICATION_CREDENTIALS). Skipping Vertex AI initialization.")

            # As a fallback, try to initialize genai client with Vertex AI if not already set and all creds are present
            if self.client is None and project_id and location and credentials_path:
                try:
                    if not self.credentials:
                        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
                        self.logger.info("Loaded service account credentials from file (fallback).")
                    vertexai.init(
                        project=project_id,
                        location=location,
                        credentials=self.credentials
                    )
                    self.logger.info("Initializing genai SDK with Vertex AI as fallback.")
                    self.client = genai.Client(
                        vertexai=True,
                        credentials=self.credentials,
                        project=project_id,
                        location=location,
                    )
                    
                    self.client_type = "genai"
                    self.logger.info(f"Successfully initialized direct genai client using model {self.model}")
                except Exception as e:
                    self.logger.warning(f"Direct GenAI initialization with Vertex AI failed: {e}. No client initialized.")
                    self.client = None
                    self.client_type = None

        except Exception as e:
            logger.error(f"Failed to initialize Google LLM Engine: {e}")
            raise

    def generate(self, messages: List[Dict[Any, Any]], **kwargs) -> str:
        """
        Generate a response from the model based on the given prompt.

        :param messages: The input prompt for the model.
        :param kwargs: Additional arguments for the model.
        :return: The generated response as a string.
        """
        if self.client_type == "vertexai":
            # For Vertex AI GenerativeModel
            # Assume messages is a list of dicts with 'content' keys
            prompt = "\n".join([msg.get("content", "") for msg in messages])
            response = self.client.generate_content(prompt, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        elif self.client_type == "genai":
            # For genai.Client
            prompt = "\n".join([msg.get("content", "") for msg in messages])
            response = self.client.generate_content(model=self.model, prompt=prompt, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        else:
            raise RuntimeError("Client not initialized properly.")
        
    def generate_from_image(self, image_bytes: List[bytes], messages: List[Dict[Any, Any]], **kwargs) -> str:
        """
        Generate a response from the model based on the content of an image.

        :param image_bytes: The bytes of the image file.
        :param messages: The input prompt for the model.
        :param kwargs: Additional arguments for the model.
        :return: The generated response as a string.
        """
        contents = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                contents.append(msg["content"])
        
        if self.client_type == "vertexai":
            # For Vertex AI GenerativeModel
            from vertexai.generative_models import Part
            image_parts = [Part.from_data(image, mime_type="image/jpeg") for image in image_bytes]
            contents.extend(image_parts)
            response = self.client.generate_content(contents=contents, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        elif self.client_type == "genai":
            # For genai.Client
            from google.genai.types import Part
            image_parts = [Part.from_bytes(image, mime_type="image/jpeg") for image in image_bytes]
            contents.extend(image_parts)
            response = self.client.generate_content(model=self.model, contents=contents, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        else:
            raise RuntimeError("Client not initialized properly.")
        
    def generate_from_pdf(self, pdf_bytes: List[bytes], messages: List[Dict[Any, Any]] ,**kwargs) -> str:
        """
        Generate a response from the model based on the content of a PDF file.

        :param pdf_bytes: The bytes of the PDF file.
        :param messages: The input prompt for the model.
        :param kwargs: Additional arguments for the model.
        :return: The generated response as a string.
        """
        contents = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                contents.append(msg["content"])
        
        if self.client_type == "vertexai":
            # For Vertex AI GenerativeModel
            from vertexai.generative_models import Part
            pdf_parts = [Part.from_data(pdf, mime_type="application/pdf") for pdf in pdf_bytes]
            contents.extend(pdf_parts)
            response = self.client.generate_content(contents=contents, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        elif self.client_type == "genai":
            # For genai.Client
            from google.genai.types import Part
            pdf_parts = [Part.from_bytes(pdf, mime_type="application/pdf") for pdf in pdf_bytes]
            contents.extend(pdf_parts)
            response = self.client.generate_content(model=self.model, contents=contents, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        else:
            raise RuntimeError("Client not initialized properly.")
        

    def __repr__(self):
        return f"GoogleLLM(model_name={self.model}, mode={self.client_type})"