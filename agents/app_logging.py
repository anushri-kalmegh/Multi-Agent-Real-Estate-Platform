"""Structured JSON logging for application events."""

import json
import logging
from datetime import datetime, timezone

from agents.config import LOG_DIR


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


def get_logger(name: str = "propwise") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(LOG_DIR / "app.jsonl", encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
