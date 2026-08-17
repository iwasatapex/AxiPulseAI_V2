"""CLI command implementations retained as thin adapters over canonical engines."""
from __future__ import annotations

import argparse

from .defaults import display_defaults, load_defaults
from .utils import bounded


def cprint(message):
    print(message)


def left_print(message):
    print(message)


def render_full_dashboard(data=None):
    if data is not None:
        print(data)
    return data


def run_train(*_args, **_kwargs):
    from train_all_ai import main as train_main
    return train_main()


def run_predict(*_args, **_kwargs):
    from forecast_cli import main as predict_main
    return predict_main()


def run_predict_nps(*_args, **_kwargs):
    from api.services.nps_service import NPSService
    return NPSService().predict(*_args, **_kwargs)


def run_reverse_nps(*_args, **_kwargs):
    from reverse_nps_solver import main as reverse_main
    return reverse_main()


def run_explain_nps(*_args, **_kwargs):
    from core.nps_predictor.explainability import explain_prediction
    return explain_prediction(*_args, **_kwargs)


def run_leaderboard_nps(*_args, **_kwargs):
    return run_predict_nps(*_args, **_kwargs)


def run_defaults(*_args, **_kwargs):
    return display_defaults(load_defaults())


def run_train_nps_predictor(*_args, **_kwargs):
    from core.nps_predictor.trainer import NPSModelTrainer
    return NPSModelTrainer(*_args, **_kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="predict_cli")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("defaults")
    args, _ = parser.parse_known_args(argv)
    if args.command == "defaults":
        return run_defaults()
    parser.print_help()
    return 0
