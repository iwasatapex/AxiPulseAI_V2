from pathlib import Path
import importlib.util


checks = {}


files = [
    "Dockerfile",
    "docker-compose.yml",
    ".github/workflows/axipulse-ci.yml",
    "monitoring/prometheus.yml",
    "api/main.py",
    "api/auth/jwt.py",
    "api/database/connection.py",
    "tests/test_api_health.py"
]


for f in files:

    checks[f] = Path(f).exists()



modules = [
    "fastapi",
    "sqlalchemy",
    "prometheus_client",
    "jose"
]


for m in modules:

    checks[
        "module_"+m
    ] = importlib.util.find_spec(m) is not None



for k,v in checks.items():

    print(
        k,
        "PASS" if v else "FAIL"
    )


failed = [
    k for k,v in checks.items()
    if not v
]


print()

if failed:

    print(
        "FAILED:",
        failed
    )

else:

    print(
        "ALL PRODUCTION CHECKS PASS"
    )
