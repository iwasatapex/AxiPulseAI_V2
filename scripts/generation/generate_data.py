"""
Compatibility facade for the historical generate_data module.

The active implementation is generate_interactive_data.
This module preserves the historical public API without duplicating
the generation logic.
"""

from .generate_interactive_data import (
    InteractiveDataGenerator,
    clear_screen,
    print_header,
    print_section,
    print_success,
    get_float_input,
    get_int_input,
    get_user_inputs,
    show_summary,
    generate_data,
    run,
    main,
)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def get_season(*args, **kwargs):
    """
    Historical compatibility helper.

    Prefer the active InteractiveDataGenerator for actual generation.
    """
    generator = InteractiveDataGenerator()
    method = getattr(generator, "get_season", None)
    if callable(method):
        return method(*args, **kwargs)

    # Preserve a safe fallback for callers that only need the symbol.
    return "NORMAL"


def week_position_modifier(*args, **kwargs):
    """
    Historical compatibility helper.
    """
    generator = InteractiveDataGenerator()
    method = getattr(generator, "week_position_modifier", None)
    if callable(method):
        return method(*args, **kwargs)
    return 1.0


def sample_complexity(*args, **kwargs):
    """
    Historical compatibility helper.
    """
    generator = InteractiveDataGenerator()
    method = getattr(generator, "sample_complexity", None)
    if callable(method):
        return method(*args, **kwargs)
    return "Medium"


def call_volume(*args, **kwargs):
    """
    Historical compatibility helper.
    """
    generator = InteractiveDataGenerator()
    method = getattr(generator, "call_volume", None)
    if callable(method):
        return method(*args, **kwargs)
    return 0


def is_exceptional(*args, **kwargs):
    """
    Historical compatibility helper.
    """
    generator = InteractiveDataGenerator()
    method = getattr(generator, "is_exceptional", None)
    if callable(method):
        return method(*args, **kwargs)
    return False


def generate_intelligence_factor(*args, **kwargs):
    """
    Historical compatibility helper.
    """
    generator = InteractiveDataGenerator()
    method = getattr(generator, "generate_intelligence_factor", None)
    if callable(method):
        return method(*args, **kwargs)
    return 1.0


def generate_weekdays(*args, **kwargs):
    """
    Historical compatibility helper.
    """
    generator = InteractiveDataGenerator()
    method = getattr(generator, "generate_weekdays", None)
    if callable(method):
        return method(*args, **kwargs)
    return []


def simulate(*args, **kwargs):
    """
    Historical compatibility entry point.

    Uses the active generator when possible.
    """
    generator = InteractiveDataGenerator()
    method = getattr(generator, "simulate", None)
    if callable(method):
        return method(*args, **kwargs)

    return generate_data(generator)


if __name__ == "__main__":
    main()
