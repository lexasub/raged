"""
__main__.py - Entry point for `python -m ast_rag`.

Delegates to the Typer CLI defined in cli.py.
"""

from ast_rag.cli import app

if __name__ == "__main__":
    app()
