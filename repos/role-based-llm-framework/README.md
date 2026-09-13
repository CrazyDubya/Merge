# ChipCliff: Role-Based Collaborative Framework

An advanced role-based conversational framework designed to facilitate collaborative problem-solving and task management in software development projects. ChipCliff orchestrates the interaction between Project Managers, Developers, and Researchers using multiple LLM providers for enhanced AI assistance.

## 🎯 Features

- **Role-Based Architecture**: Dedicated workflows for Project Manager, Developer, and Researcher personas
- **Multi-LLM Integration**: Supports OpenAI, Anthropic Claude, and DeepSeek models
- **Intelligent Task Classification**: Automatically categorizes tasks as coding or research-oriented
- **Real-time Collaboration**: WebSocket-based notifications for instant team updates
- **Secure Configuration**: Environment variable-based API key management
- **Modular Design**: Clean separation of concerns with dedicated algorithms per role
- **Web Interface**: FastAPI-powered dashboard with role-specific views

## 🏗️ Architecture

### Core Components

- **`main.py`**: FastAPI application entry point with routing and middleware
- **`pm_algorithm.py`**: Project management logic and task orchestration
- **`coder_algorithm.py`**: Code generation, testing, and development workflows
- **`researcher_algorithm.py`**: Research task handling and information gathering
- **`llm_integration.py`**: Multi-provider LLM abstraction layer
- **`xml_utils.py`**: XML-based data persistence and retrieval
- **`ui.py`**: WebSocket connection management and real-time updates
- **`utils.py`**: Configuration loading and utility functions

### Role Workflows

#### Project Manager
- Task classification and assignment
- Progress tracking and coordination
- Resource allocation and timeline management
- Cross-role communication facilitation

#### Developer
- Code generation and implementation
- Automated testing and validation
- Technical architecture decisions
- Code review and optimization

#### Researcher
- Information gathering and analysis
- Requirements research and documentation
- Technology evaluation and recommendations
- Market and competitive analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for at least one LLM provider (OpenAI, Anthropic, or DeepSeek)

### Installation

1. **Clone and setup**:
   ```bash
   git clone https://github.com/CrazyDubya/chipcliff-collaborative-framework.git
   cd chipcliff-collaborative-framework
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. **Run the application**:
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the interface**:
   Open `http://localhost:8000` in your browser

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your API credentials:

```bash
# Required: At least one LLM provider API key
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional: Application settings
DEBUG=false
LOG_LEVEL=INFO
FLASK_SECRET_KEY=your_secret_key_here
```

### Role Configuration

The framework supports three primary roles, each with specific capabilities:

- **Project Manager**: Task coordination, resource management, timeline planning
- **Developer**: Code implementation, testing, technical architecture
- **Researcher**: Information gathering, analysis, documentation

## 🔧 API Usage

### Task Classification Endpoint

```python
POST /classify-task
{
    "description": "Build a user authentication system",
    "context": "Web application development"
}
```

### Role-Specific Chat

```python
POST /chat/{role}
{
    "message": "How should we implement OAuth2?",
    "context": "Authentication system"
}
```

### Real-time Updates

WebSocket endpoint for live collaboration:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    // Handle real-time updates
};
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_pm_algorithm.py
python -m pytest tests/test_coder_algorithm.py
python -m pytest tests/test_researcher_algorithm.py
```

## 📁 Project Structure

```
chipcliff-collaborative-framework/
├── config/
│   └── config.yaml              # Application configuration
├── static/
│   ├── css/                     # Stylesheets
│   ├── js/                      # JavaScript files
│   └── images/                  # Static images
├── templates/
│   ├── base.html               # Base template
│   ├── index.html              # Main dashboard
│   ├── pm_dashboard.html       # Project Manager view
│   ├── coder_dashboard.html    # Developer view
│   └── researcher_dashboard.html # Researcher view
├── tests/                       # Unit tests
├── logs/                        # Application logs
├── main.py                     # FastAPI application
├── pm_algorithm.py             # Project management logic
├── coder_algorithm.py          # Development workflows
├── researcher_algorithm.py     # Research workflows
├── llm_integration.py          # LLM provider integration
├── xml_utils.py                # Data persistence
├── ui.py                       # WebSocket handling
├── utils.py                    # Utility functions
└── requirements.txt            # Python dependencies
```

## 🔒 Security

- **API Key Management**: All credentials stored in environment variables
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Secure error responses without sensitive information
- **CORS Configuration**: Configurable cross-origin resource sharing

## 🚀 Deployment

### Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Performance

- **Async Architecture**: FastAPI with async/await for concurrent request handling
- **WebSocket Optimization**: Efficient real-time communication
- **LLM Connection Pooling**: Optimized API calls to language model providers
- **Caching**: Intelligent caching of research results and code snippets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Use type hints for better code clarity

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for modern web API development
- Integrates with leading LLM providers for AI-powered assistance
- Inspired by collaborative software development methodologies

## 📞 Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the test files for usage examples