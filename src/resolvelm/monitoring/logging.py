from __future__ import annotations

import logging
import sys
from typing import Any

try:
    from pythonjsonlogger.jsonlogger import JsonFormatter
except ImportError:  # python-json-logger >= 3.2
    from pythonjsonlogger.json import JsonFormatter


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.info(message, extra=fields)
