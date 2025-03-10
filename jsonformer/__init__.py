"""
JSONFormer is a library for structured JSON generation using language models.
This module serves as the main entry point for the library, exposing the core
components needed by users.
"""

# Import the main Jsonformer class which handles JSON generation
from jsonformer.main import Jsonformer
# Import utility function for highlighting JSON values in output
from jsonformer.format import highlight_values