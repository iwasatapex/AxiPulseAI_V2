from pathlib import Path
import os


checks = {}


# Secrets

checks["jwt_secret_configured"] = bool(
    os.getenv(
        "AXIPULSE_JWT_SECRET"
    )
)


checks["api_key_configured"] = bool(
    os.getenv(
        "AXIPULSE_API_KEY"
    )
)



# Files

files = [

    "api/auth/jwt.py",

    "api/security/api_key.py",

    "api/auth/dependencies.py",

    "api/main.py",

    "deployment/.env.example"

]


for f in files:

    checks[
        "file_"+f
    ] = Path(f).exists()



# Security structure

checks[
    "authentication_layer"
] = Path(
    "api/auth"
).exists()


checks[
    "security_layer"
] = Path(
    "api/security"
).exists()



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
        "SECURITY ISSUES:",
        failed
    )

else:

    print(
        "SECURITY REVIEW PASS"
    )
