import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

import time
import statistics

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def benchmark(
    endpoint,
    method="GET",
    count=100
):

    results = []

    for _ in range(count):

        start = time.perf_counter()

        if method == "GET":

            client.get(endpoint)

        else:

            client.post(
                endpoint,
                json={
                    "state":{
                        "timeline":[
                            {
                                "operations_health":82,
                                "competency":88,
                                "quality":85,
                                "attendance":90,
                                "release":60,
                                "transfer":14,
                                "nps":88
                            }
                        ]
                    },
                    "recommendations":[]
                }
            )

        results.append(
            time.perf_counter() - start
        )


    return {
        "requests": count,
        "avg_ms":
            round(
                statistics.mean(results)
                * 1000,
                3
            ),
        "p95_ms":
            round(
                statistics.quantiles(
                    results,
                    n=20
                )[-1]
                * 1000,
                3
            )
    }



if __name__ == "__main__":

    print(
        "HEALTH",
        benchmark("/health")
    )

    print(
        "ADIE",
        benchmark(
            "/api/v1/adie/decision",
            "POST",
            50
        )
    )
