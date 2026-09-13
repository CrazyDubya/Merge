"""
Main application with improved error handling and configuration.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import logging
import os
import uvicorn
from typing import Optional

# Import modules with error handling
try:
    from llm_integration import call_openai, call_anthropic, call_deepseek
except ImportError as e:
    logging.warning(f"LLM integration modules not available: {e}")
    call_openai = call_anthropic = call_deepseek = None

try:
    from xml_utils import create_xml_schema
except ImportError as e:
    logging.warning(f"XML utils not available: {e}")
    create_xml_schema = None

try:
    from pm_algorithm import classify_task, assign_task, update_status
except ImportError as e:
    logging.warning(f"PM algorithm not available: {e}")
    classify_task = assign_task = update_status = None

try:
    from coder_algorithm import generate_code, send_feedback
except ImportError as e:
    logging.warning(f"Coder algorithm not available: {e}")
    generate_code = send_feedback = None

try:
    from researcher_algorithm import generate_queries
except ImportError as e:
    logging.warning(f"Researcher algorithm not available: {e}")
    generate_queries = None

try:
    from ui import create_ui, setup_dashboards, setup_notifications
except ImportError as e:
    logging.warning(f"UI modules not available: {e}")
    create_ui = setup_dashboards = setup_notifications = None

try:
    from utils import load_configuration, handle_error, setup_logging
except ImportError as e:
    logging.warning(f"Utils module not available: {e}")
    load_configuration = handle_error = setup_logging = None

# Initialize FastAPI app
app = FastAPI(
    title="ChipCliff Role-Based LLM Framework",
    description="An advanced role-based conversational framework for collaborative problem-solving",
    version="1.0.0",
    debug=os.getenv("DEBUG", "false").lower() == "true"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logger = logging.getLogger(__name__)

# Load configuration safely
config = {}
if load_configuration:
    try:
        config = load_configuration()
        if setup_logging:
            setup_logging(config)
    except Exception as e:
        logger.warning(f"Could not load configuration: {e}")

# Define data models
class ChatRequest(BaseModel):
    prompt: str
    
    class Config:
        str_strip_whitespace = True

class TaskRequest(BaseModel):
    description: str
    context: Optional[str] = None
    
    class Config:
        str_strip_whitespace = True

# Configuration for models
models_config = {}
if call_openai:
    models_config["openai"] = call_openai
if call_anthropic:
    models_config["anthropic"] = call_anthropic
if call_deepseek:
    models_config["deepseek"] = call_deepseek

@app.get("/")
async def read_root():
    """Root endpoint returning application info."""
    return {
        "message": "Welcome to ChipCliff Role-Based LLM Framework",
        "version": "1.0.0",
        "available_models": list(models_config.keys()),
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_available": len(models_config),
        "configuration_loaded": bool(config)
    }

@app.post("/chat/{model_name}")
async def chat(model_name: str, request: ChatRequest):
    """Chat endpoint for LLM interaction."""
    if model_name not in models_config:
        available_models = ", ".join(models_config.keys()) if models_config else "none"
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available models: {available_models}"
        )
    
    try:
        call_model = models_config[model_name]
        response = call_model(request.prompt)
        logger.info(f"Successful chat response from {model_name}")
        return JSONResponse(content={
            "response": response, 
            "model": model_name,
            "status": "success"
        })
    except Exception as err:
        logger.error(f"Error in chat with {model_name}: {err}")
        raise HTTPException(status_code=500, detail=f"Error calling {model_name}: {str(err)}")

@app.post("/task")
async def handle_task(request: TaskRequest):
    """Handle task classification and assignment."""
    if not classify_task or not assign_task:
        raise HTTPException(
            status_code=503, 
            detail="Task handling services not available"
        )
    
    try:
        # Use local LLM for classification
        category = classify_task(request.description)
        if not category:
            raise HTTPException(status_code=500, detail="Failed to classify task")
        
        task_id = assign_task(category, request.description)
        if not task_id:
            raise HTTPException(status_code=500, detail="Failed to assign task")
        
        return JSONResponse(content={
            "task_id": task_id, 
            "category": category,
            "status": "assigned",
            "description": request.description
        })
    except ValueError as e:
        logger.error(f"Task handling error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as err:
        logger.error(f"Unexpected error handling task: {err}")
        if handle_error:
            handle_error(err, "task handling")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get status of a specific task."""
    # This would need to be implemented with proper data retrieval
    return JSONResponse(content={
        "task_id": task_id,
        "status": "pending",  # Placeholder
        "message": "Task status retrieval not fully implemented"
    })

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input data", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    if handle_error:
        handle_error(exc, "global handler")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred", 
            "type": type(exc).__name__
        }
    )

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting ChipCliff Role-Based LLM Framework")
    
    # Initialize XML schema if available
    if create_xml_schema:
        try:
            create_xml_schema()
        except Exception as e:
            logger.warning(f"Could not initialize XML schema: {e}")
    
    # Initialize UI components if available
    if create_ui and setup_dashboards and setup_notifications:
        try:
            create_ui()
            setup_dashboards()
            setup_notifications()
        except Exception as e:
            logger.warning(f"Could not initialize UI components: {e}")

@app.on_event("shutdown") 
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down ChipCliff Role-Based LLM Framework")
def main():
    """Main function for direct execution."""
    
    # Setup basic logging if not already configured
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    logger.info("Starting ChipCliff application...")
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )

if __name__ == "__main__":
    main()
