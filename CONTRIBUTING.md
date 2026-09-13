# Contributing to Village

Thank you for your interest in contributing to the Village project! This document provides guidelines and information for contributors.

## 🌟 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read and follow our [Code of Conduct](#code-of-conduct-details).

## 🚀 Quick Start for Contributors

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/yourusername/Village.git`
3. **Create a feature branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes**
5. **Test your changes**: `pytest`
6. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- pip or poetry

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/CrazyDubya/Village.git
cd Village

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest
```

## 📝 Contribution Types

We welcome various types of contributions:

### 🐛 Bug Reports
- Use the bug report template
- Include reproduction steps
- Provide system information
- Add relevant logs or screenshots

### ✨ Feature Requests
- Use the feature request template
- Explain the use case
- Describe the expected behavior
- Consider implementation challenges

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test additions

### 📚 Documentation
- API documentation
- Usage examples
- Architecture guides
- Tutorial improvements

## 📋 Contribution Guidelines

### Code Style

We follow PEP 8 with some modifications:

```python
# Use type hints
def create_villager(name: str, role: str) -> Villager:
    return Villager(name=name, role=role)

# Use descriptive variable names
villager_count = len(village.villagers)
llm_response = provider.generate(prompt)

# Use docstrings for all public functions
def collaborate(self, task: str) -> str:
    """Execute a collaborative task across all villagers.
    
    Args:
        task: The task description to be executed
        
    Returns:
        The collaborative result as a string
        
    Raises:
        VillageError: If no villagers are available
    """
```

### Commit Messages

Follow conventional commits format:

```
type(scope): description

- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting changes
- refactor: code refactoring
- test: adding tests
- chore: maintenance tasks

Examples:
feat(villager): add memory persistence capability
fix(llm): handle rate limiting errors gracefully
docs(readme): update installation instructions
```

### Branch Naming

Use descriptive branch names:
- `feature/add-claude-provider`
- `fix/memory-leak-in-village`
- `docs/update-api-reference`
- `refactor/simplify-communication`

## 🧪 Testing Guidelines

### Test Types

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

### Writing Tests

```python
# Unit test example
def test_villager_creation():
    """Test villager can be created with valid parameters."""
    villager = Villager(name="test", role="analyst")
    assert villager.name == "test"
    assert villager.role == "analyst"

# Integration test example
def test_village_villager_interaction():
    """Test village can manage villagers properly."""
    village = Village()
    villager = Villager(name="test", role="analyst")
    
    village.add_villager(villager)
    assert len(village.villagers) == 1
    assert village.get_villager("test") == villager
```

### Test Coverage

- Aim for >90% test coverage
- Focus on critical paths first
- Include edge cases and error conditions
- Test both success and failure scenarios

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=village --cov-report=html

# Run specific test file
pytest tests/unit/test_villager.py

# Run tests matching pattern
pytest -k "test_memory"

# Run tests in parallel
pytest -n auto
```

## 🔍 Code Review Process

### For Contributors

1. **Self-review**: Review your own code before submitting
2. **Description**: Provide clear PR description
3. **Tests**: Include appropriate tests
4. **Documentation**: Update relevant documentation
5. **Changelog**: Add entry if user-facing change

### Review Criteria

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it readable and maintainable?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security concerns?
- **Tests**: Are there adequate tests?
- **Documentation**: Is documentation updated?

### Review Timeline

- **Initial Response**: Within 2 business days
- **Full Review**: Within 1 week
- **Re-review**: Within 2 business days after updates

## 📦 Release Process

### Versioning

We use Semantic Versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git
- [ ] Publish to PyPI

## 🎯 Project Priorities

### Current Focus Areas

1. **Core Architecture**: Stable foundation for village interactions
2. **LLM Providers**: Support for major LLM services
3. **Documentation**: Comprehensive guides and examples
4. **Testing**: Robust test coverage
5. **Performance**: Optimization for concurrent operations

### Future Roadmap

- Visual village designer
- Distributed village deployment
- Advanced analytics
- Machine learning optimization

## 🆘 Getting Help

### Resources

- **Documentation**: [docs/](docs/)
- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Discord**: [Village Community Server](https://discord.gg/village-ai)

### Mentorship

New contributors can request mentorship:
- Comment on "good first issue" tickets
- Join our Discord for real-time help
- Attend virtual office hours (TBD)

## 🏷️ Issue Labels

- `good first issue`: Good for newcomers
- `help wanted`: Community help needed
- `bug`: Bug reports
- `enhancement`: Feature requests
- `documentation`: Documentation improvements
- `question`: General questions
- `security`: Security-related issues

## 📊 Recognition

### Contributors

All contributors are recognized in:
- AUTHORS.md file
- Release notes
- Annual contributor highlights

### Hall of Fame

Special recognition for:
- Significant feature contributions
- Major bug fixes
- Outstanding community support
- Security vulnerability reports

## 🔐 Security Considerations

- Never include API keys in commits
- Review security implications of changes
- Report security issues privately
- Follow secure coding practices

## 📞 Contact

- **Project Maintainers**: maintainers@village-ai.com
- **Community**: community@village-ai.com
- **Security**: security@village-ai.com

## Code of Conduct Details

### Our Pledge

We pledge to make participation in our project and community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Harassment, trolling, or discriminatory comments
- Publishing others' private information without permission
- Personal or political attacks
- Other conduct inappropriate in a professional setting

### Enforcement

Instances of unacceptable behavior may be reported to the project team at conduct@village-ai.com. All reports will be reviewed and investigated promptly and fairly.

---

*Thank you for contributing to Village! 🏘️*