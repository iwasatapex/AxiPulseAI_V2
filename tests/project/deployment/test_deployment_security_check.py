import importlib

def test_security_check_imports():
    importlib.import_module("deployment.security_check")
