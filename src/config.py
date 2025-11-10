import os
from dotenv import load_dotenv

load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
# DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
# OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gemini‑2.5‑flash')

# Add to your config.py:
MODEL_PROVIDER = "gemini"  # or "openai"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # or reuse OPENAI_API_KEY
GEMINI_MODEL = "gemini-2.0-flash-001" 
OPENAI_MODEL = "gpt-4"  # your OpenAI model
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')