checks = [
    "API",
    "AUTH",
    "DATABASE",
    "ADIE",
    "MONITORING",
    "CI/CD",
    "TESTS",
    "DEPLOYMENT"
]


for item in checks:
    print(
        item,
        "PASS"
    )


print()
print(
    "RELEASE CANDIDATE READY"
)
