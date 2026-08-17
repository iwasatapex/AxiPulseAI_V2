def get_input(prompt, cast=float, default=None):
    value = input(prompt)
    if not value and default is not None:
        return default
    return cast(value)


def collect_day_inputs():
    fields = (
        "quality",
        "competency",
        "attendance",
        "release",
        "transfer",
        "operations_health",
    )
    return {field: get_input(f"{field}: ") for field in fields}
