import sys
import time
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# Logger setup

logger.remove()


# Request logs (clean one-line output)

logger.add(
    sys.stdout,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[method]}</cyan> <cyan>{extra[path]}</cyan> "
        "→ <level>{extra[status]}</level> "
        "({extra[latency_ms]}ms)"
    ),
    filter=lambda rec: "method" in rec["extra"],
    colorize=True,
)


# General logs

logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    filter=lambda rec: "method" not in rec["extra"],
    colorize=True,
)


# File logs (JSON)

logger.add(
    "logs/api.log",
    level="INFO",
    rotation="10 MB",
    retention=5,
    serialize=True,
    enqueue=True,
)


# Public logger

log = logger


# Middleware

class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        status = response.status_code
        method = request.method
        path = request.url.path

        level = "INFO"
        if status >= 500:
            level = "ERROR"
        elif status >= 400:
            level = "WARNING"

        logger.bind(
            method=method,
            path=path,
            status=status,
            latency_ms=latency_ms,
        ).log(level, "")

        return response