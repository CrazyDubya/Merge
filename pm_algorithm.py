import os
import uuid
from typing import Optional, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from xml_utils import update_task_status, log_error, log_success
from coder_algorithm import generate_code, send_feedback
from researcher_algorithm import generate_queries, store_results

# Directory to store the model and tokenizer
MODEL_DIR = "models/task_classifier"
MODEL_NAME = "distilbert-base-uncased"

def load_model_and_tokenizer() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
        except Exception as e:
            log_error(f"Failed to download or save model/tokenizer: {str(e)}")
            raise SystemExit(e)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def classify_task(user_input: str) -> Optional[str]:
    try:
        # This function uses a local LLM (DistilBERT) for classification
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        category = "coding" if predictions.item() == 0 else "research"
        return category
    except Exception as e:
        log_error(f"Error during task classification: {str(e)}")
        return None

def assign_task(category: str, task_description: str) -> Optional[str]:
    task_id = str(uuid.uuid4())
    try:
        if category == 'coding':
            code, test_result = generate_code(task_description)
            send_feedback(test_result)
        elif category == 'research':
            summary = generate_queries(task_description)
            store_results(summary)
        else:
            raise ValueError(f"Unknown category: {category}")
        
        update_status(task_id, "completed")
        log_success(task_id)
        return task_id
    except Exception as e:
        log_error(f"Error assigning task {task_id}: {str(e)}")
        return None

def update_status(task_id: str, status: str) -> None:
    try:
        update_task_status(task_id, status)
        print(f"Task {task_id} status updated to {status}")
    except Exception as e:
        log_error(f"Error updating status for task {task_id}: {str(e)}")

if __name__ == "__main__":
    user_input = "Create a new HTML page with interactive elements"
    category = classify_task(user_input)
    if category:
        task_id = assign_task(category, user_input)
        if task_id:
            update_status(task_id, "In Progress")
