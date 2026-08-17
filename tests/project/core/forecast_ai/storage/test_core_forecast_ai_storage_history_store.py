import importlib

def test_history_store_surface():
    module = importlib.import_module("core.forecast_ai.storage.history_store")
    assert hasattr(module, "load")
    assert hasattr(module, "save")
    assert hasattr(module, "append")
    assert hasattr(module, "HistoryStore")
