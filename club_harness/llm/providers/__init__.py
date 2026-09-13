"""Optional LLM provider backends ported from Village.

These are an *optional extra* - Merge core stays httpx-only. Provider SDKs
(openai, anthropic, google-generativeai) are imported lazily inside each
module; importing this package never requires them. See
requirements-village.txt for the extra.

Ported from Village (repos/Village) via git subtree merge.
Provenance: docs/PROVENANCE.md.
"""

from .base import BaseLLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider

__all__ = ["BaseLLMProvider", "OpenAIProvider", "AnthropicProvider", "GoogleProvider"]
