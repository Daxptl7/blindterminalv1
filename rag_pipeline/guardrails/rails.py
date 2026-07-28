"""
rails.py — NeMo Guardrails Orchestrator
==========================================
Initializes the NeMo Guardrails engine and registers
custom RAG actions (retrieve_context, generate_textbook_response).
Acts as the safety and routing layer before queries reach the LLM.
"""

import logging
from typing import Optional
from .colang_rules import get_colang_rules, get_nemo_config

logger = logging.getLogger("GuardrailsRails")

class GuardrailsEngine:
    """
    Manages the NeMo Guardrails engine lifecycle.
    Routes queries through safety checks before executing RAG actions.
    """

    def __init__(self, retriever_fn=None, generator_fn=None):
        """
        Args:
            retriever_fn: Callable(query: str) -> str that retrieves relevant context.
            generator_fn: Callable(query: str, context: str) -> str that generates a response.
        """
        self.retriever_fn = retriever_fn
        self.generator_fn = generator_fn
        self.rails = None
        self._initialized = False

        self._init_nemo()

    def _init_nemo(self):
        """Attempts to initialize NeMo Guardrails with the defined configuration."""
        try:
            from nemoguardrails import LLMRails, RailsConfig

            colang_content = get_colang_rules()
            config_dict = get_nemo_config()

            config = RailsConfig.from_content(
                colang_content=colang_content,
                yaml_content=config_dict
            )
            self.rails = LLMRails(config)

            # Register custom RAG actions
            if self.retriever_fn:
                self.rails.register_action(self.retriever_fn, name="retrieve_context")

            if self.generator_fn:
                self.rails.register_action(self.generator_fn, name="generate_textbook_response")

            self._initialized = True
            logger.info("NeMo Guardrails engine initialized successfully.")

        except ImportError:
            logger.warning(
                "nemoguardrails package not installed. "
                "Install with: pip install nemoguardrails. "
                "Guardrails will be bypassed."
            )
        except Exception as e:
            logger.error(f"Failed to initialize NeMo Guardrails: {e}")

    @property
    def is_active(self) -> bool:
        """Check if NeMo Guardrails is initialized and active."""
        return self._initialized and self.rails is not None

    async def process_query_async(self, user_message: str) -> str:
        """
        Routes a user query through NeMo Guardrails (async version).
        If guardrails are not available, falls back to direct RAG execution.

        Args:
            user_message: The user's raw query text.

        Returns:
            The guardrail-checked response string.
        """
        if not self.is_active:
            return self._fallback_query(user_message)

        try:
            response = await self.rails.generate_async(prompt=user_message)
            return response
        except Exception as e:
            logger.error(f"Guardrails async processing error: {e}")
            return self._fallback_query(user_message)

    def process_query(self, user_message: str) -> str:
        """
        Routes a user query through NeMo Guardrails (sync version).
        If guardrails are not available, falls back to direct RAG execution.

        Args:
            user_message: The user's raw query text.

        Returns:
            The guardrail-checked response string.
        """
        if not self.is_active:
            return self._fallback_query(user_message)

        try:
            response = self.rails.generate(prompt=user_message)
            return response
        except Exception as e:
            logger.error(f"Guardrails processing error: {e}")
            return self._fallback_query(user_message)

    def _fallback_query(self, user_message: str) -> str:
        """
        Direct RAG execution without guardrails.
        Used as a fallback when NeMo is not installed or initialization failed.
        """
        if not self.retriever_fn or not self.generator_fn:
            return "RAG pipeline is not configured. Please check your setup."

        try:
            context = self.retriever_fn(query=user_message)
            if not context or not context.strip():
                return "I couldn't find relevant information in the textbooks for your question."
            response = self.generator_fn(query=user_message, context=context)
            return response
        except Exception as e:
            logger.error(f"Fallback query error: {e}")
            return "I'm sorry, I encountered an issue processing your question."
