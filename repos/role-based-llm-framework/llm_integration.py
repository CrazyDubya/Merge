# llm_integration.py
import requests
import json
import anthropic
from typing import Dict, Any
from utils import load_configuration

config = load_configuration()

def call_openai(prompt: str) -> str:
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['openai_api_key']}"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    except requests.RequestException as e:
        raise Exception(f"Error calling OpenAI API: {str(e)}")

def call_anthropic(prompt: str) -> str:
    try:
        client = anthropic.Client(config['anthropic_api_key'])
        response = client.completions.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens_to_sample=1000,
            temperature=0,
            prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
            stop_sequences=["\n\nHuman:"]
        )
        return response.completion
    except anthropic.APIError as e:
        raise Exception(f"Error calling Anthropic API: {str(e)}")

def call_deepseek(prompt: str) -> str:
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['deepseek_api_key']}"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    except requests.RequestException as e:
        raise Exception(f"Error calling DeepSeek API: {str(e)}")
