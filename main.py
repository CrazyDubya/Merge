from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from llm_integration import call_openai, call_anthropic, call_deepseek
from xml_utils import create_xml_schema
from pm_algorithm import classify_task, assign_task, update_status
from coder_algorithm import generate_code, send_feedback
from researcher_algorithm import generate_queries
from ui import create_ui, setup_dashboards, setup_notifications
from utils import load_configuration, handle_error

# Initialize FastAPI app
app = FastAPI(debug=True)

# Load configuration
config = load_configuration()

# Initialize UI
create_ui()
setup_dashboards()
setup_notifications()

# Define data models
class ChatRequest(BaseModel):
    prompt: str

class TaskRequest(BaseModel):
    description: str

# Configuration for models
models_config = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "deepseek": call_deepseek,
}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Role-Based Conversational Framework"}

@app.post("/chat/{model_name}")
async def chat(model_name: str, request: ChatRequest):
    if model_name not in models_config:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        call_model = models_config[model_name]
        response = call_model(request.prompt)
        logger.info(f"Success: {response}")
        return JSONResponse(content={"response": response, "model": model_name})
    except Exception as err:
        logger.error(f"Error occurred: {err}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/task")
async def handle_task(request: TaskRequest):
    try:
        # Use local LLM for classification
        category = classify_task(request.description)
        task_id = assign_task(category, request.description)
        
        if category == 'coding':
            # Send coding task to external model
            code, test_result = generate_code(request.description)
            send_feedback(test_result)
        elif category == 'research':
            # Send research task to external model
            summary = generate_queries(request.description)
        else:
            raise ValueError(f"Unknown category: {category}")
        
        update_status(task_id, "completed")
        return JSONResponse(content={"task_id": task_id, "status": "completed", "category": category})
    except Exception as err:
        logger.error(f"Error handling task: {err}")
        raise HTTPException(status_code=500, detail="Error handling task")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"An error occurred: {str(exc)}")
    handle_error(exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred", "type": type(exc).__name__}
    )

def main():
    create_xml_schema()
    # Additional initialization steps can be added here

if __name__ == "__main__":
    main()
