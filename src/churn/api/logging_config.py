from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

# Campos emitidos pelo LogRecord que não fazem parte do contexto da aplicação.
# São excluídos do repasse via "extra" para evitar ruído nos logs.
_STDLIB_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Formata registros de log como JSON de linha única para ingestão estruturada."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        log_obj: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Inclui campos extras injetados via logger.xxx(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _STDLIB_ATTRS and not key.startswith("_"):
                log_obj[key] = value

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_obj["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_obj, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configura o logger raiz e o logger do pacote ``churn`` com saída em JSON.

    Deve ser chamada uma única vez na inicialização da aplicação (ex: dentro de ``create_app``).
    Chamadas subsequentes são idempotentes pois ``force=True`` substitui handlers existentes
    no logger raiz.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logging.basicConfig(
        level=level.upper(),
        handlers=[handler],
        force=True,
    )

    # Silencia loggers de terceiros excessivamente verbosos
    for noisy in ("uvicorn.access", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
