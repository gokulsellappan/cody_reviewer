import time
import logging
import os
import sys
from openai import OpenAI
import google.generativeai as genai
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Explicit stdout handler
        # Uncomment below to also log to file:
        # logging.FileHandler('app.log')
    ]
)

logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.ERROR)


# ✨ NEW: Model provider configuration
MODEL_PROVIDER = getattr(config, "MODEL_PROVIDER", "openai").lower()  # "openai" or "gemini"

# ✨ CHANGED: Separate API keys and models for each provider
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_MODEL = getattr(config, "OPENAI_MODEL", "gpt-4")

GEMINI_API_KEY = getattr(config, "GEMINI_API_KEY", config.OPENAI_API_KEY)  # Fallback to OpenAI key

GEMINI_MODEL = getattr(config, "GEMINI_MODEL", "gemini-2.0-flash-001")

# ✨ NEW: Initialize clients based on provider
if MODEL_PROVIDER == "openai":
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info(f"Initialized OpenAI client with model: {OPENAI_MODEL}")
elif MODEL_PROVIDER == "gemini":
    genai.configure(api_key=GEMINI_API_KEY)
    logging.info(f"Initialized Gemini client with model: {GEMINI_MODEL}")
else:
    logging.warning(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}, defaulting to OpenAI")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_PROVIDER = "openai"

def query_openai(prompt: str, retries=3, base_delay=1.0) -> str:
    """
    Send a prompt to the configured AI model (OpenAI or Gemini) and return the response.
    
    ✨ CHANGED: Now supports both OpenAI and Gemini based on MODEL_PROVIDER config
    
    Args:
        prompt: The prompt to send to the AI model
        retries: Number of retry attempts on failure
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        The AI model's response text or an error message
    """
    attempt = 0
    while attempt < retries:
        try:
            # ✨ NEW: Route to appropriate provider
            if MODEL_PROVIDER == "gemini":
                return _query_gemini(prompt)
            else:
                return _query_openai(prompt)
                
        except Exception as e:
            logging.error(f"{MODEL_PROVIDER.upper()} API error on attempt {attempt + 1}: {e}")
            attempt += 1
            
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.critical(f"Final attempt failed with error: {e}")

    error_message = f"Failed to query {MODEL_PROVIDER.upper()} API after {retries} attempts."
    logging.error(error_message)
    return error_message

# ✨ NEW: Separate function for OpenAI queries
def _query_openai(prompt: str) -> str:
    """Internal function to query OpenAI API."""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a senior software engineer at Google reviewing a pull request."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
    )
    
    if response.choices and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    else:
        return "No response content from OpenAI."

# ✨ NEW: Separate function for Gemini queries
def _query_gemini(prompt: str) -> str:
    """Internal function to query Gemini API."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Gemini accepts a list of content parts or a string
    full_prompt = (
        "You are a senior software engineer at Google reviewing a pull request.\n\n"
        f"{prompt}"
    )
    
    response = model.generate_content(full_prompt)
    
    # ✨ CHANGED: Better error handling for Gemini responses
    if response and hasattr(response, "text") and response.text:
        return response.text.strip()
    elif response and hasattr(response, "candidates") and response.candidates:
        # Check if response was blocked
        if hasattr(response.candidates[0], "finish_reason"):
            finish_reason = response.candidates[0].finish_reason
            if finish_reason != 1:  # 1 = STOP (normal completion)
                return f"Gemini response blocked or incomplete. Reason code: {finish_reason}"
        return "No valid text in Gemini response."
    else:
        return "No response from Gemini API."