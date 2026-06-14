"""
HyDE (Hypothetical Document Embedding) tool.
Generates a synthetic answer to the query, embeds it, and uses that vector
for retrieval — dramatically improves recall for vague/conceptual questions.
"""
from __future__ import annotations

import logging

from shared.azure_clients import get_openai_client
from shared.config import settings

logger = logging.getLogger(__name__)


def generate_hypothetical_document(query: str) -> str:
    """
    Prompt the LLM to write a plausible answer passage.
    We deliberately keep temperature slightly above zero for diversity.
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable enterprise assistant. "
                    "Write a concise, factual passage (3-5 sentences) that directly answers the question below. "
                    "Do not add caveats, disclaimers, or 'I'. "
                    "Write as if this text would appear in an internal policy document."
                ),
            },
            {"role": "user", "content": f"Question: {query}"},
        ],
        temperature=0.4,
        max_tokens=300,
    )
    hypothetical = response.choices[0].message.content.strip()
    logger.debug("HyDE generated document (first 120 chars): %s", hypothetical[:120])
    return hypothetical
