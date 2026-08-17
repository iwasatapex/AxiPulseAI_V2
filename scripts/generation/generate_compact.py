"""
Compatibility facade for the historical compact generator.

The canonical implementation is generate_interactive_data.
"""

from .generate_interactive_data import (
    Colors,
    clear_screen,
    print_header,
    print_section,
    print_success,
    get_float_input,
    get_user_inputs,
    generate_data,
    show_summary,
    run,
    main,
    InteractiveDataGenerator,
)


def get_float(prompt, default=0.0, min_val=None, max_val=None):
    return get_float_input(
        prompt,
        default,
        min_val=min_val,
        max_val=max_val,
    )


def display_stats(generator):
    method = getattr(generator, "display_statistics", None)
    if callable(method):
        return method()

    method = getattr(generator, "calculate_statistics", None)
    if callable(method):
        return method()

    return None


def save_files(generator, *args, **kwargs):
    method = getattr(generator, "save_data", None)
    if callable(method):
        return method(*args, **kwargs)
    return None


def get_mode(*args, **kwargs):
    """
    Historical compact-generator input surface.

    The active generator has a single generation flow, so NORMAL is
    used as the compatibility default.
    """
    return "NORMAL"


class CompactGenerator(InteractiveDataGenerator):
    """
    Backward-compatible class name.

    Generation behavior remains inherited from the active generator.
    """
    pass


if __name__ == "__main__":
    main()
