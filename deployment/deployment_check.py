from pathlib import Path
import importlib.util


checks = {}


files = [

    "Dockerfile",

    "docker-compose.yml",

    ".env",

    "deployment/start.sh",

    "deployment/healthcheck.sh",

    "monitoring/prometheus.yml",

    ".github/workflows/axipulse-ci.yml",

    "api/main.py",

    "api/auth/jwt.py",

    "api/database/connection.py",

    "release/version.py"

]


for f in files:

    checks[
        f
    ] = Path(f).exists()



modules = [

    "fastapi",

    "sqlalchemy",

    "prometheus_client",

    "jose",

    "pytest"

]


for m in modules:

    checks[
        "dependency_"+m
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
        "DEPLOYMENT BLOCKERS:",
        failed
    )

else:

    print(
        "PRODUCTION DEPLOYMENT READY"
    )
