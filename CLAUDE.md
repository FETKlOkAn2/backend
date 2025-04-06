# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands
- Install dependencies: `pip install -r requirements.txt`
- Type checking: `mypy core/ --config-file mypy.ini`
- Run main app: `python -m core`
- Run GUI: `python gui/crypto_trader.py`

## Code Style Guidelines
- **Imports**: Standard libs first, third-party next, project modules last
- **Type Annotations**: Use type hints in function signatures; mypy validation is recommended but optional
- **Naming**: 
  - Classes: PascalCase (RSI, Strategy)
  - Methods/Functions: snake_case (calculate_rsi)
  - Variables: snake_case (current_symbol)
- **Error Handling**: Use try/except blocks with specific context in error messages, log errors with "error" type
  - Example: `self.log(f"Error processing {symbol}: {str(e)}", "error")`
- **Documentation**: Use docstrings for classes and functions
- **Formatting**: 4 spaces for indentation
- **Performance**: Use NumPy and numba (@njit) for performance-critical code

When making changes, match the existing code style in the file you're modifying.