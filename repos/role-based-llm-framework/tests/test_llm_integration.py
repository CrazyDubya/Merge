import pytest
from ChipCliff.llm_integration import call_openai, call_anthropic, call_deepseek
from ChipCliff.utils import load_configuration

config = load_configuration()

@pytest.mark.skipif(not config.get('openai_api_key'), reason="OpenAI API key not found")
def test_call_openai():
    response = call_openai("Say this is a test!")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.skipif(not config.get('anthropic_api_key'), reason="Anthropic API key not found")
def test_call_anthropic():
    response = call_anthropic("Why is the ocean salty?")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.skipif(not config.get('deepseek_api_key'), reason="DeepSeek API key not found")
def test_call_deepseek():
    response = call_deepseek("Hello!")
    assert isinstance(response, str)
    assert len(response) > 0
