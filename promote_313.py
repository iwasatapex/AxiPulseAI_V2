import sys
sys.path.insert(0, ".")
from core.forecast_ai.prediction.production_registry import register_production, MANIFEST_NAME
import json
# Promote the rebuilt 1mil-10yr family to canonical production through the
# hardened registry (validates role, loadability, provenance, atomic activation).
r = register_production("1mil-10yr")
print("promoted:", r["source_family"], "->", r["family"], "gen:", r.get("generation"))
print("oh_path:", r["oh_path"])
print("nps_path:", r["nps_path"])
