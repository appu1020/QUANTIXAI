import os
import logging
from django.conf import settings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = os.path.join(settings.BASE_DIR, '.env')
load_dotenv(env_path)

# Verify environment variable loading
api_key = os.getenv("GROQ_API_KEY")
logger.info("GROQ_API_KEY Loaded: %s", bool(api_key))

def get_groq_client():
    """Initialize and return the Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is missing from .env file!")
        return None
        
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        logger.info("Groq Client Initialized: True")
        return client
    except ImportError:
        logger.error("Groq SDK is not installed. Run `pip install groq`.")
        return None
    except Exception as e:
        logger.error("Failed to initialize Groq client: %s", e)
        return None

def generate_rag_response(
    prompt: str,
    model: str = None,
    temperature: float = 0.2
) -> str:
    """
    Generate an answer using Groq Cloud API using the fully constructed hybrid prompt.
    """
    client = get_groq_client()
    if not client:
        return "Error: LLM API key or client is missing. Please configure GROQ_API_KEY."

    # Model selection logic
    PREFERRED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "qwen/qwen3-14b",
        "gemma2-9b-it"
    ]
    
    selected_model = model
    if not selected_model:
        try:
            available_models = [m.id for m in client.models.list().data]
            for pref_model in PREFERRED_MODELS:
                if pref_model in available_models:
                    selected_model = pref_model
                    break
            if not selected_model and available_models:
                selected_model = available_models[0]
        except Exception as e:
            logger.warning("Failed to fetch models from Groq: %s", e)
            selected_model = "llama-3.3-70b-versatile"
            
    logger.info("Selected Groq Model: %s", selected_model)
    logger.debug("Prompt length: %d chars", len(prompt))

    try:
        start_time = __import__('time').time()
        
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=800,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        response_time = __import__('time').time() - start_time
        logger.info("Groq API Response Time: %.2f seconds", response_time)
        
        answer = completion.choices[0].message.content.strip()
        logger.info("Groq successfully generated response (%d chars).", len(answer))
        return answer
        
    except Exception as e:
        logger.error("Groq API Error: %s", e)
        # Return the actual exception to the UI for debugging purposes
        return f"API Error: {str(e)}"

