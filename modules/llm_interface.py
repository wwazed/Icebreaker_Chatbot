"""Module for interfacing with IBM watsonx.ai LLMs."""

import logging

from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM

import config

logger = logging.getLogger(__name__)


def create_watsonx_embedding() -> WatsonxEmbeddings:
    """Create an IBM watsonx embedding model.

    Returns:
        Configured WatsonxEmbeddings model.
    """
    watsonx_embedding = WatsonxEmbeddings(
        model_id=config.EMBEDDING_MODEL_ID,
        url=config.WATSONX_URL,
        project_id=config.WATSONX_PROJECT_ID,
        truncate_input_tokens=3,
    )

    logger.info(
        f"Created Watsonx Embedding model: {config.EMBEDDING_MODEL_ID}"
    )
    return watsonx_embedding


def create_watsonx_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample"
) -> WatsonxLLM:
    """Create an IBM watsonx LLM for generating responses.

    Args:
        temperature: Temperature for controlling generation randomness.
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: Decoding method to use ("sample" or "greedy").

    Returns:
        Configured WatsonxLLM model.
    """
    additional_params = {
        "decoding_method": decoding_method,
        "min_new_tokens": config.MIN_NEW_TOKENS,
        "top_k": config.TOP_K,
        "top_p": config.TOP_P,
    }

    watsonx_llm = WatsonxLLM(
        model_id=config.LLM_MODEL_ID,
        url=config.WATSONX_URL,
        project_id=config.WATSONX_PROJECT_ID,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        additional_params=additional_params,
    )

    logger.info(f"Created Watsonx LLM model: {config.LLM_MODEL_ID}")
    return watsonx_llm


def change_llm_model(new_model_id: str) -> None:
    """Update the model ID used when creating an LLM.

    Args:
        new_model_id: New LLM model ID to use.
    """
    config.LLM_MODEL_ID = new_model_id

    logger.info(f"Changed LLM model to: {new_model_id}")