from typing import List, Dict, Any
import pandas as pd
import io
from src.core.extractor import BaseExtractor

import logging
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ExcelExtractor(BaseExtractor):
    def extract(self, file_bytes: bytes, as_markdown: bool = True) -> str:
        """
        Convert Excel file (from bytes) to string representation.

        Args:
            file_bytes (bytes): Bytes of the Excel file.
            as_markdown (bool): If True, format as markdown table; else plain text.

        Returns:
            str: Formatted text representation of Excel file.
        """
        try:
            xls = pd.ExcelFile(io.BytesIO(file_bytes))
            result = []

            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)

                df = df.astype(str)
                df = df.replace("nan", "", regex=False)  # Replace "nan" with empty string

                result.append(f"### Sheet: {sheet_name}\n")

                if as_markdown:
                    table_str = df.to_markdown(index=False)
                else:
                    table_str = df.to_string(index=False)

                result.append(table_str + "\n")
            logger.info("Excel extraction completed successfully.")
        except Exception as e:
            logger.error(f"Failed to extract text from Excel file: {e}")
            raise RuntimeError(f"Failed to extract text from Excel file: {e}")
        
        return "\n".join(result)