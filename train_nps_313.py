import sys
sys.path.insert(0, ".")
from core.nps_predictor import NPSPredictor
nps = NPSPredictor()
print("NPS: training on 1mil-10yr.csv ...", flush=True)
nps.train("training/1mil-10yr.csv")
nps.save_model("models/1mil-10yr_NPS.pkl")
print("NPS training complete, model:", nps.model_name, "feats:", len(nps.feature_names), flush=True)
