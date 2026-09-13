# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-01-14 (Phase 1 Complete)

### Added
- **OpenAI Provider**: Full integration with GPT-3.5, GPT-4, and GPT-4 Turbo
  - Async/await support
  - Streaming responses
  - Context window awareness (4K-128K tokens)
  - Function calling support detection
- **Anthropic Provider**: Full integration with Claude models
  - Claude 3 (Opus, Sonnet, Haiku) support
  - Claude 2 and Instant support
  - Large context windows (100K-200K tokens)
  - Vision capabilities (Claude 3)
- **Google Gemini Provider**: Full integration with Gemini models
  - Gemini Pro and Pro Vision support
  - Gemini 1.5 Pro/Flash support
  - Massive context windows (up to 1M tokens)
  - Safety settings configuration
- **Persistent Storage System**:
  - Abstract storage interface
  - In-memory storage for development
  - PostgreSQL storage with async connection pooling
  - Village, villager, memory, and task history persistence
  - Automatic table creation and migrations
- **Rate Limiting**: Token bucket algorithm implementation
  - Configurable requests per minute
  - Burst size support
  - Blocking and non-blocking token acquisition
- **Quota Management**: Multi-period quota tracking
  - Hourly, daily, and monthly quotas
  - Rolling window tracking
  - Usage statistics
- **Prometheus Metrics**: Comprehensive monitoring
  - LLM request metrics (count, duration, tokens)
  - Villager and village metrics
  - Storage operation metrics
  - Rate limiting and quota metrics
  - Error tracking by component
- **Phase 1 Documentation**:
  - PHASE1_FEATURES.md with complete feature documentation
  - Enhanced examples demonstrating new capabilities
  - Updated requirements and dependencies
- **Comprehensive Tests**:
  - Provider unit tests
  - Storage backend tests
  - Rate limiting tests
  - Metrics tests

### Changed
- Updated requirements.txt with new dependencies
- Updated pyproject.toml with optional dependencies
- Enhanced LLM provider __init__.py with conditional imports
- Improved error handling across all providers

### Performance
- Async connection pooling for database operations
- Efficient token bucket rate limiting
- Optimized metric collection and export
- Streaming support for real-time LLM responses

### Security
- Environment variable-based API key management
- Secure PostgreSQL connection string handling
- Rate limiting to prevent API abuse
- Quota management for cost control

## [0.1.0] - 2024-12-22

### Added
- Initial project structure and architecture
- Core Village and Villager classes
- LLM provider abstraction layer
- Basic memory management system
- Inter-villager communication framework
- Comprehensive test suite
- Security policy and contribution guidelines
- Development workflow and CI/CD setup
- Documentation structure

### Security
- Input validation framework
- Secure API key handling guidelines
- Security policy for vulnerability reporting

## [0.1.0] - 2024-12-22

### Added
- Initial release with basic village architecture
- Support for multiple LLM providers
- Collaborative task execution
- Memory and communication systems
- Comprehensive documentation and examples