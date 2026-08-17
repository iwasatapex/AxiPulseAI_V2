
class ForecastCalibration:

    def summarize(
        self,
        error_history
    ):

        if not error_history:
            return {
                "samples": 0,
                "status": "NO_DATA"
            }

        avg_error = sum(
            x["error"]
            for x in error_history
        ) / len(error_history)

        return {
            "samples": len(error_history),
            "average_error": avg_error,
            "status": "CALIBRATED"
        }
