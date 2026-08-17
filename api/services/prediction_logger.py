from api.services.persistence_service import PersistenceService


class PredictionLogger:


    def __init__(self):

        self.storage = PersistenceService()



    def log(
        self,
        model,
        inputs,
        output
    ):

        return self.storage.save_prediction(
            model,
            inputs,
            output
        )

# Module-level compatibility surface
def log(model, inputs, output):
    return PredictionLogger().log(model, inputs, output)
