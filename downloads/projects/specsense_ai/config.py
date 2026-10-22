import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")   # from .env - never hard-code

# LLM Settings
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")   # e.g. gemini-2.5-flash or gemini-3.5-flash-lite
MAX_RETRIES = 3
RETRY_DELAY = 2

# Processing Settings
DEFAULT_BATCH_SIZE = 10
STREAMLIT_BATCH_SIZE = 5
