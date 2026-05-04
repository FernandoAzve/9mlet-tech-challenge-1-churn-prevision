"""Compatibilidade: use ``from churn.logging_config import ...``."""

from churn.logging_config import JsonFormatter, configure_logging

__all__ = ["JsonFormatter", "configure_logging"]
