# Contributing

We welcome contributions to AutoSlide! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/autoslide.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

This includes:

- pytest for testing
- black for code formatting
- isort for import sorting
- flake8 for linting
- pre-commit hooks

## Code Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 88 characters (Black default)
- Use type hints where appropriate
- Write docstrings for all public functions

### Formatting

Format code with Black:

```bash
black autoslide/
```

Sort imports with isort:

```bash
isort autoslide/
```

### Linting

Run flake8:

```bash
flake8 autoslide/
```

## Testing

Write tests for new features:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=autoslide tests/
```

## Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

This automatically runs formatting and linting before each commit.

## Documentation

Update documentation when adding features:

1. Add docstrings to new functions
2. Update relevant markdown files in `docs/`
3. Add examples if applicable
4. Update API reference if needed

Build documentation locally:

```bash
mkdocs serve
```

## Pull Request Guidelines

- **Clear description** - Explain what changes you made and why
- **Reference issues** - Link to related issues
- **Tests** - Include tests for new features
- **Documentation** - Update docs as needed
- **Small PRs** - Keep changes focused and manageable
- **Clean history** - Squash commits if necessary

## Reporting Issues

When reporting bugs, include:

- AutoSlide version
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

## Feature Requests

For feature requests, describe:

- The problem you're trying to solve
- Your proposed solution
- Alternative approaches considered
- Potential impact on existing functionality

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

- Open an issue for questions
- Check existing issues and documentation first
- Provide context and examples

Thank you for contributing to AutoSlide!
