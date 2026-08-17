import sys
sys.path.insert(0, ".")
from core.operation_health_predictor import OperationalHealthPredictor
oh = OperationalHealthPredictor()
print("OH: training on 1mil-10yr.csv ...", flush=True)
oh.train("training/1mil-10yr.csv")
oh.save_model("models/1mil-10yr_OH.pkl")
print("OH training complete, model:", oh.model_name, "feats:", len(oh.feature_names), flush=True)
