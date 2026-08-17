import os


required = [
    "AXIPULSE_API_KEY",
    "AXIPULSE_JWT_SECRET",
]


missing = []

for key in required:

    if not os.getenv(key):
        missing.append(key)


if missing:

    print(
        "MISSING:",
        missing
    )

else:

    print(
        "ENV SECURITY PASS"
    )
