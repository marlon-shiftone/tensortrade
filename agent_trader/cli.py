from __future__ import annotations

import argparse

from .integrations.alpaca.cli import register_parser as register_alpaca_parser
from .integrations.kucoin.cli import register_parser as register_kucoin_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control plane multi-broker para bots traders.")
    integrations = parser.add_subparsers(dest="integration", required=True)
    register_alpaca_parser(integrations)
    register_kucoin_parser(integrations)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("nenhum handler associado ao comando selecionado")
        return 2
    return handler(args)
