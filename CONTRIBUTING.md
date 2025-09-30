# Contributing to Quant Bot

Thank you for your interest in contributing to Quant Bot! This document provides guidelines and best practices for contributing.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional but recommended)
- Git
- Basic understanding of algorithmic trading and Python

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/quant-bot.git
   cd quant-bot
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Setup Environment**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

4. **Run Tests**
   ```bash
   make test
   ```

### Docker Development (Recommended)

```bash
# Start development container
make shell

# Inside container, run tests
pytest -v

# Run linters
make lint
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test additions or fixes

### 2. Make Changes

Follow these guidelines:

#### Code Style

- Use [Black](https://black.readthedocs.io/) for formatting (line length 100)
- Use [Ruff](https://docs.astral.sh/ruff/) for linting
- Follow PEP 8 conventions
- Use type hints for all functions
- Write docstrings for public APIs

```python
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        periods: Number of periods per year (252 for daily)
        
    Returns:
        Annualized Sharpe ratio
        
    Example:
        >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
    """
    excess_returns = returns - (risk_free_rate / periods)
    return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()
```

#### Testing

- Write tests for all new features
- Maintain or improve code coverage (target: >=90%)
- Use pytest for testing
- Mock external API calls

```python
def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    sharpe = calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert sharpe > 0
```

#### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic changes)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

Examples:
```
feat(data): add support for Polygon.io data provider
fix(backtest): correct position sizing calculation
docs(readme): update installation instructions
test(validate): add tests for timezone handling
```

### 3. Run Quality Checks

Before committing:

```bash
# Format code
black .

# Lint code
ruff check . --fix

# Type check (optional but encouraged)
mypy config data_providers data_tools storage

# Run tests
pytest -v

# Check coverage
make test.cov
```

### 4. Submit Pull Request

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

3. **PR Description Should Include:**
   - Clear description of changes
   - Related issue numbers (if any)
   - Screenshots (for UI changes)
   - Testing performed
   - Breaking changes (if any)

## What to Contribute

### Good First Issues

Look for issues labeled `good first issue` or `help wanted`.

### Areas Needing Contribution

1. **Data Providers**
   - Add new data provider integrations
   - Improve existing provider reliability

2. **Testing**
   - Increase test coverage
   - Add integration tests
   - Add property-based tests with Hypothesis

3. **Documentation**
   - Improve existing docs
   - Add tutorials and examples
   - Create video guides

4. **Features**
   - New trading strategies
   - ML model improvements
   - Discord bot enhancements

5. **Performance**
   - Optimize data processing
   - Improve backtest speed
   - Add async support

### What We're NOT Looking For

- Live trading features (system is paper-only by design)
- Cryptocurrency wallet integrations
- Payment processing features
- Breaking changes without discussion

## Review Process

1. **Automated Checks**
   - CI must pass (tests, linting, type checking)
   - Coverage must not decrease

2. **Code Review**
   - At least one maintainer approval required
   - Address all review comments
   - Keep PR focused and small

3. **Merge**
   - Maintainer will merge when approved
   - Squash merge is default

## Development Guidelines

### Architecture Principles

1. **Separation of Concerns**
   - Data providers in `data_providers/`
   - Business logic in `apps/` and `algos/`
   - Storage in `storage/`
   - UI in `ui/`

2. **Configuration**
   - Use `config.settings` for all configuration
   - Never hardcode credentials
   - Use environment variables

3. **Error Handling**
   - Use specific exception types
   - Log errors with context
   - Never expose secrets in error messages

4. **Logging**
   - Use loguru for logging
   - Use appropriate log levels
   - Include context in log messages

### Security

- Never commit secrets or API keys
- Use `Settings.masked_dict()` when logging config
- Validate all user inputs
- Use parameterized queries (via ORM)
- Follow [SECURITY.md](SECURITY.md) guidelines

### Performance

- Profile before optimizing
- Use vectorized operations (NumPy, Pandas)
- Consider memory usage for large datasets
- Cache expensive computations

## Communication

- **Issues:** Bug reports, feature requests, questions
- **Discussions:** General topics, ideas, support
- **Pull Requests:** Code contributions

## Questions?

- Check existing [Issues](https://github.com/KaholiK/quant-bot/issues)
- Read the [Documentation](README.md)
- Ask in [Discussions](https://github.com/KaholiK/quant-bot/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Quant Bot!** 🚀
