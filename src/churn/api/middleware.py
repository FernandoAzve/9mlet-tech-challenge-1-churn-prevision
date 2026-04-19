from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware que:

    * Atribui um ID de correlação a cada requisição (header ``X-Request-ID``).
    * Mede a latência fim a fim da resposta em milissegundos.
    * Emite uma linha de log JSON estruturado por requisição com rota, método,
      código de status, duração e o ID de correlação.
    * Reflete ambos os valores ao chamador via headers de resposta.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Respeita o ID de correlação fornecido pelo chamador; caso contrário, gera um novo.
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            logger.exception(
                "unhandled_exception",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "route": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                },
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 3)

        logger.info(
            "request_finished",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(duration_ms)

        return response
