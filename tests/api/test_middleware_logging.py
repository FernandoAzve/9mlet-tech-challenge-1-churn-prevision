from __future__ import annotations

import json
import logging
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

from churn.logging_config import JsonFormatter, configure_logging
from churn.api.middleware import LatencyLoggingMiddleware


def test_latency_middleware_sets_headers() -> None:
    app = FastAPI()
    app.add_middleware(LatencyLoggingMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/ping", headers={"X-Request-ID": "req-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-123"
    assert "X-Process-Time-Ms" in response.headers


def test_latency_middleware_handles_exception() -> None:
    app = FastAPI()
    app.add_middleware(LatencyLoggingMiddleware)

    @app.get("/boom")
    def boom():
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500


def test_json_formatter_includes_extra_fields() -> None:
    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-456"

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "hello"
    assert payload["request_id"] == "req-456"


def test_json_formatter_includes_exception_and_stack() -> None:
    formatter = JsonFormatter()

    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=2,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )
    record.stack_info = "stack"

    payload = json.loads(formatter.format(record))

    assert "exception" in payload
    assert payload["stack_info"] == "stack"


def test_configure_logging_sets_noisy_loggers() -> None:
    configure_logging("INFO")

    assert logging.getLogger("uvicorn.access").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING
