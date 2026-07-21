"""CLI demo for the LLM-based Time Series Assistant prototype."""

from __future__ import annotations

import argparse

from agent import run_agent
from utils import load_sample_data


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="LLM-based Time Series Assistant (sktime prototype)")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Natural language question, e.g. 'forecast next 12 steps'",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start a simple interactive session.",
    )
    return parser


def run_interactive() -> None:
    """Run an interactive text loop."""
    data = load_sample_data()
    print("LLM-based Time Series Assistant")
    print("Type 'exit' to quit.")
    print("Examples: forecast next 12 steps | what is the trend? | detect anomalies")

    while True:
        query = input("\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not query:
            continue

        answer = run_agent(query, data)
        print(answer)


def main() -> None:
    """Entry point for CLI execution."""
    args = build_parser().parse_args()

    if args.interactive:
        run_interactive()
        return

    data = load_sample_data()

    # If no query is provided, run a tiny scripted demo.
    if args.query is None:
        demo_queries = [
            "forecast next 12 steps",
            "what is the trend?",
            "detect anomalies",
        ]
        for q in demo_queries:
            print("=" * 80)
            print(run_agent(q, data))
        return

    print(run_agent(args.query, data))


if __name__ == "__main__":
    main()
