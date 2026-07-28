import os
import logging
from google import genai
from google.genai import types
from PIL import Image
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("GeminiAgentService")

class GeminiAgent:
    """
    Interfaces with Google Gemini API to provide contextual reasoning for RAG.
    Defaults to the latest configured Gemini Flash model.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-3.5-flash'):
        # Fallback to env var if key not explicitly passed
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key or key.startswith("YOUR_"):
            logger.warning("No valid Gemini API key provided. Gemini Agent will operate in mock mode.")
            self.client = None
            return

        try:
            self.client = genai.Client(api_key=key)
            self.model_name = model_name
            self.system_instruction = (
                "You are BlindAssist, a supportive, clear, and concise AI assistant for visually impaired students. "
                "You will be provided with NCERT textbook context extracted via RAG. "
                "Use the provided context to answer questions accurately in 2-3 simple sentences suitable for speech synthesis. "
                "If the context does not contain the answer, state honestly that the information wasn't found in the text."
            )
            logger.info(f"Gemini Agent successfully initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Agent: {e}")
            self.client = None

    def describe_image(self, pil_image: Image.Image, context_hint: str = "") -> str:
        """
        Uses Gemini's multimodal capability to generate a detailed text explanation
        of a textbook diagram, chart, or illustration.
        """
        #if gemini online model is present then it goes here 
        if not self.client:
            return "Image description is unavailable (offline mode)."
        prompt = (
            "You are an expert textbook illustrator and educational assistant for visually impaired students. "
            "Analyze this diagram, chart, flow-chart, or illustration. "
            "Explain in detail what is happening, including all text labels, equations, arrows, "
            "and scientific relations shown in the image. Be clear and descriptive."
        )
        if context_hint:
            prompt += f"\nContext hint: This image is captioned/titled '{context_hint}'."
        try:
            # Pass both the prompt text and the PIL image object
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, pil_image]
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.error(f"Error describing image: {e}")
        return ""

    def generate_response(self, query: str, context: str) -> str:
        """
        Generates a contextual response based on user query and retrieved RAG context.
        """
        if not query:
            return "No question received."

        if not self.client:
            return f"[Offline Mock Response] Based on context: '{context[:100]}...', here is the answer for '{query}'."

        # Construct augmented prompt
        prompt = f"NCERT Document Context:\n{context}\n\nUser Question: {query}"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction
                )
            )
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            return "I'm sorry, I couldn't generate a clear answer from the document."
        except Exception as e:
            logger.error(f"Gemini API Error during generation: {e}")
            return f"Gemini service encountered an error: {e}"
