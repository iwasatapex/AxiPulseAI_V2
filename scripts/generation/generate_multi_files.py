"""
Compatibility facade for the historical multi-file generator.

The canonical generation implementation is generate_interactive_data.
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


def get_float_input_compat(
    prompt,
    default=0.0,
    min_val=None,
    max_val=None,
):
    return get_float_input(
        prompt,
        default,
        min_val=min_val,
        max_val=max_val,
    )


def get_user_inputs_compat(*args, **kwargs):
    return get_user_inputs(*args, **kwargs)


def save_files(generator, *args, **kwargs):
    method = getattr(generator, "save_data", None)
    if callable(method):
        return method(*args, **kwargs)
    return None


def show_summary_compat(generator, *args, **kwargs):
    return show_summary(generator, *args, **kwargs)


class MultiFileGenerator(InteractiveDataGenerator):
    """
    Backward-compatible class name.

    Generation behavior remains inherited from the active generator.
    """
    pass


# Historical names expected by callers/tests.
get_float_input_original = get_float_input
get_user_inputs_original = get_user_inputs
show_summary_original = show_summary

# Keep the exact historical public names.
def get_float_input(prompt, default=0.0, min_val=None, max_val=None):
    return get_float_input_original(
        prompt,
        default,
        min_val=min_val,
        max_val=max_val,
    )


def get_user_inputs(*args, **kwargs):
    return get_user_inputs_original(*args, **kwargs)


def show_summary(generator, *args, **kwargs):
    return show_summary_original(generator, *args, **kwargs)


if __name__ == "__main__":
    main()
