import time

from starlette.middleware.base import BaseHTTPMiddleware

from api.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ACTIVE_REQUESTS
)


class MetricsMiddleware(
    BaseHTTPMiddleware
):


    async def dispatch(
        self,
        request,
        call_next
    ):

        start = time.time()

        ACTIVE_REQUESTS.inc()

        response = await call_next(request)

        elapsed = time.time() - start


        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code)
        ).inc()


        REQUEST_LATENCY.labels(
            endpoint=request.url.path
        ).observe(elapsed)


        ACTIVE_REQUESTS.dec()


        return response

    
# Module-level compatibility surface
dispatch = MetricsMiddleware.dispatch
