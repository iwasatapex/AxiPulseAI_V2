from pathlib import Path
import hashlib


files = [
    "api/main.py",
    "Dockerfile",
    "docker-compose.yml",
    "release/version.py"
]


print("RELEASE FREEZE FILES")


for f in files:

    path = Path(f)

    if path.exists():

        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()[:12]

        print(
            f,
            "FROZEN",
            digest
        )

    else:

        print(
            f,
            "MISSING"
        )


print()
print(
    "VERSION FREEZE COMPLETE"
)
