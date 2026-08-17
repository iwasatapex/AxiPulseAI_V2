from locust import (
    HttpUser,
    task,
    between
)


class AxiPulseUser(HttpUser):

    wait_time = between(
        1,
        3
    )


    @task
    def health(self):

        self.client.get(
            "/health"
        )


    @task
    def system_status(self):

        self.client.get(
            "/api/v1/system/status"
        )


    @task
    def metrics(self):

        self.client.get(
            "/metrics"
        )


    @task
    def adie(self):

        self.client.post(
            "/api/v1/adie/decision",
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
