"""
Core Connectors - Connect to existing core modules
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core modules
try:
    from core.operation_health_predictor.predictor import OperationHealthPredictor
    from core.nps_predictor.predictor import NPSPredictor
except ImportError:
    # Core modules not available, will use mock services
    pass
