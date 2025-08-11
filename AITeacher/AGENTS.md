# AGENTS.md

## Build, Lint, and Test Commands
- Install dependencies: `pip install -r requirements.txt`
- Run all tests: `pytest`
- Run a single test: `pytest path/to/test_file.py::TestClass::test_method`
- Lint code: `ruff .`

## Code Style Guidelines
- Follow PEP8: 4 spaces, no tabs, max 79 chars/line.
- Import order: standard libraries, third-party, then local modules.
- Use explicit imports; avoid wildcard imports.
- Add type hints for all functions and methods.
- Class names: CamelCase. Function/variable names: snake_case.
- Constants: ALL_CAPS.
- Handle errors with try/except; log or raise meaningful exceptions.
- Remove unused variables and imports.
- Keep functions short, focused, and well-documented.
- Write docstrings for all public classes and functions (describe purpose, args, returns).
- No Cursor or Copilot rules detected in this repository.
