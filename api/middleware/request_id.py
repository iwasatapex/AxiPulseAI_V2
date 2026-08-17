import uuid
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(
    BaseHTTPMiddleware
):

    async def dispatch(
        self,
        request,
        call_next
    ):

        request_id = str(uuid.uuid4())

        response = await call_next(request)

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response

    
# Module-level compatibility surface
dispatch = RequestIDMiddleware.dispatch
