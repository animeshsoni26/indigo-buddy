# Contributing to IndiGo Buddy

Thank you for your interest in contributing to IndiGo Buddy! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit bug fixes
- ✨ Add new features
- 🧪 Write tests
- 🎨 Improve UI/UX

## 🚀 Getting Started

### 1. Fork the Repository

Click the "Fork" button at the top right of the repository page.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/indigo-buddy.git
cd indigo-buddy
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

- **Line Length:** 88 characters (Black default)
- **Quotes:** Double quotes for strings
- **Imports:** Organized using isort
- **Type Hints:** Required for all public functions

#### Format Your Code

```bash
# Format with Black
black indigo_buddy.py

# Check with flake8
flake8 indigo_buddy.py

# Type check with mypy
mypy indigo_buddy.py
```

### Naming Conventions

- **Functions:** `lowercase_with_underscores`
- **Classes:** `PascalCase`
- **Constants:** `UPPERCASE_WITH_UNDERSCORES`
- **Private methods:** `_leading_underscore`

### Documentation

#### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed. Explain what the function does,
    any important details, and edge cases.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Example:
        >>> function_name("test", 5)
        True
    """
    pass
```

#### Comment Guidelines

- Use comments sparingly - prefer self-documenting code
- Explain *why*, not *what*
- Keep comments up-to-date with code changes

```python
# Good
# Use exponential backoff to avoid overwhelming the API
retry_delay *= 2

# Bad
# Multiply retry_delay by 2
retry_delay *= 2
```

### Testing

#### Writing Tests

```python
import pytest
from indigo_buddy import classify_question, get_relevant_policy

def test_classify_baggage_question():
    """Test that baggage questions are classified as policy."""
    question = "What is the baggage allowance?"
    assert classify_question(question) == 'policy'

def test_classify_service_question():
    """Test that service questions are classified as experience."""
    question = "Is the staff helpful?"
    assert classify_question(question) == 'experience'

def test_get_relevant_policy_baggage():
    """Test policy retrieval for baggage questions."""
    question = "How much luggage can I carry?"
    policy = get_relevant_policy(question)
    assert "7kg" in policy
    assert "15kg" in policy
```

#### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=indigo_buddy

# Run specific test
pytest tests/test_classification.py::test_classify_baggage_question
```

## 🔄 Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

#### Examples

```bash
feat(prompts): add hybrid prompt template

Added new prompt template that combines policy and experience
information for complex queries.

Closes #42

---

fix(response): remove prompt artifacts from output

Fixed issue where instruction text was appearing in responses
by improving the cleaning regex patterns.

Fixes #38

---

docs(readme): update installation instructions

Added section on GPU support and clarified AWS setup steps.
```

### Commit Best Practices

- Keep commits focused and atomic
- Write clear, descriptive messages
- Reference issues when applicable
- Don't commit sensitive data (.env files, credentials)

## 🔍 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Branch is up-to-date with main

### Creating a Pull Request

1. **Push Your Branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open PR on GitHub**
   - Go to the repository
   - Click "Pull Requests" → "New Pull Request"
   - Select your branch
   - Fill out the PR template

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring
   
   ## Testing
   Describe testing performed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] No breaking changes
   
   ## Related Issues
   Closes #[issue_number]
   ```

4. **Respond to Review Feedback**
   - Address all comments
   - Push updates to the same branch
   - Re-request review when ready

### Review Process

1. **Automated Checks** - CI/CD pipeline runs
2. **Code Review** - Maintainer reviews code
3. **Feedback** - Address any requested changes
4. **Approval** - PR approved by maintainer
5. **Merge** - PR merged into main branch

## 🐛 Bug Reports

### Before Reporting

- Check if bug already reported
- Verify it's reproducible
- Collect relevant information

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.9.7]
- IndiGo Buddy Version: [e.g., 2.0.0]

**Screenshots**
If applicable

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why this feature is needed

**Proposed Solution**
How you envision this working

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Mockups, examples, etc.
```

## 🎯 Priority Areas

We're especially interested in contributions in these areas:

### High Priority
- 🌐 Web interface (Flask/Streamlit)
- 🧪 Comprehensive test suite
- 🌍 Multi-language support
- 📊 Enhanced analytics

### Medium Priority
- 🎨 UI/UX improvements
- 📱 Mobile responsiveness
- 🔧 Configuration options
- 📖 Tutorial content

### Lower Priority
- 🎤 Voice interface
- 🔗 Third-party integrations
- 🤖 Additional AI models
- 🎨 Theming options

## 📚 Resources

### Documentation
- [README](README.md) - Main documentation
- [PROMPTS](PROMPTS.md) - Prompt engineering guide
- [API Reference](docs/API.md) - API documentation

### Learning Resources
- [Python Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FAISS Documentation](https://faiss.ai/)

## 🤝 Community

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person
- Assume good intentions

### Getting Help

- 💬 **Discussions** - Use GitHub Discussions for questions
- 🐛 **Issues** - Use GitHub Issues for bugs
- 📧 **Email** - Contact maintainers for sensitive issues

## 🏆 Recognition

Contributors will be:
- Listed in README contributors section
- Mentioned in release notes
- Given credit in commits

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## ❓ Questions?

If you have questions about contributing:
1. Check existing documentation
2. Search closed issues/PRs
3. Ask in GitHub Discussions
4. Contact maintainers

---

Thank you for contributing to IndiGo Buddy! 🎉
