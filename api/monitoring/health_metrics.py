from api.monitoring.metrics import (
    ENGINE_CALLS
)


def track_engine(
    name
):

    ENGINE_CALLS.labels(
        engine=name
    ).inc()
