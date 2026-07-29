"""
Shared run configuration for the scripts in ``scripts/``.

The sample end date is PINNED by default so that every shipped artifact
regenerates bit-for-bit from the parquet cache without touching the network.
Opt in to a live window explicitly:

    python scripts/run_full_analysis.py --end 2026-06-01
    python scripts/run_full_analysis.py --end live      # today − 3 days
    PM_END=live python scripts/run_full_analysis.py
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

#: End date the shipped results/ artifacts were generated with.
PINNED_END = "2026-05-04"


def resolve_end_date(default: str = PINNED_END) -> str:
    """
    Resolve the sample end date, in precedence order:

    1. ``--end YYYY-MM-DD`` or ``--end live`` on the command line
    2. ``PM_END`` environment variable (same values)
    3. the pinned default (2026-05-04, matching the shipped artifacts)

    ``live`` resolves to (today − 3 days) in Europe/Berlin — the old default,
    which made every run date-dependent and results irreproducible.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--end", default=os.environ.get("PM_END", default))
    args, _ = parser.parse_known_args()
    if args.end == "live":
        return (
            pd.Timestamp.now(tz="Europe/Berlin").normalize() - pd.Timedelta(days=3)
        ).strftime("%Y-%m-%d")
    return str(args.end)
