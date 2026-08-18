import importlib

def test_production_audit_imports():
    importlib.import_module("deployment.production_audit")
