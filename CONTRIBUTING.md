# Contributing to Indian Address Parser

Thanks for your interest in contributing! This document outlines how to get involved.

## Reporting Bugs

Open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Sample address that causes the issue (if applicable)
- Python version and OS

## Suggesting Features

Open an issue describing:
- The use case you're trying to solve
- How the feature would work
- Any alternative approaches you've considered

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run the test suite and linters
5. Commit with a clear message
6. Push to your fork and open a PR

### Development Setup

```bash
git clone https://github.com/kushagra/indian-address-parser.git
cd indian-address-parser
pip install -e ".[dev]"
pre-commit install
```

### Code Style

We use automated tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

All PRs must pass these checks. The pre-commit hooks will run them automatically.

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=address_parser --cov-report=html

# Run specific test
pytest tests/test_pipeline.py -v
```

### Commit Messages

Write clear, concise commit messages:
- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Reference issues when applicable (`Fixes #123`)

## Code Review

All submissions require review. Maintainers will provide feedback and may request changes before merging.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
