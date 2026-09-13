import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from xml_utils import store_results, log_error
from llm_integration import call_openai

def generate_queries(task_details: str) -> str:
    base_queries = [f"{task_details} best practices", f"{task_details} tutorials"]
    advanced_queries = consult_llm_for_queries(task_details, base_queries)
    results = fetch_data(advanced_queries)
    return summarize_results(results)

def fetch_data(queries: List[str]) -> List[Dict[str, str]]:
    data = []
    for query in queries:
        try:
            search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('div', class_='tF2Cxc')
            for result in results:
                title = result.find('h3').text if result.find('h3') else 'No Title'
                snippet = result.find('div', class_='VwiC3b').text if result.find('div', class_='VwiC3b') else 'No Description'
                data.append({"title": title, "description": snippet})
        except requests.RequestException as e:
            log_error(f"Failed to fetch data for query '{query}': {str(e)}")
        except Exception as e:
            log_error(f"Failed to parse data for query '{query}': {str(e)}")
    return data

def summarize_results(data: List[Dict[str, str]]) -> str:
    if not data:
        return "No results found."
    summary = "\n".join([f"Title: {item['title']}\nDescription: {item['description']}\n" for item in data[:5]])
    store_results(summary)
    return summary

def consult_llm_for_queries(task_details: str, base_queries: List[str]) -> List[str]:
    prompt = f"Given the task: '{task_details}' and base queries: {base_queries}, generate 3 more specific and targeted search queries."
    try:
        response = call_openai(prompt)
        enhanced_queries = response.split('\n')
        return base_queries + enhanced_queries
    except Exception as e:
        log_error(f"Failed to generate enhanced queries: {str(e)}")
        return base_queries
