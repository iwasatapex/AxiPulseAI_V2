from api.services.persistence_service import PersistenceService


class EnginePersistence:

    def __init__(self):
        self.storage = PersistenceService()


    def save_adie_decision(
        self,
        user,
        decision
    ):

        return self.storage.save_decision(
            user,
            decision
        )


    def save_model_prediction(
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
def save_adie_decision(user, decision):
    return EnginePersistence().save_adie_decision(user, decision)

def save_model_prediction(model, inputs, output):
    return EnginePersistence().save_model_prediction(model, inputs, output)
