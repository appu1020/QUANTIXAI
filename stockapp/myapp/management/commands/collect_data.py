"""
management/commands/collect_data.py — Scheduled market data collection.

Usage:
    python manage.py collect_data
    python manage.py collect_data --symbols AAPL,MSFT,GOOGL
    python manage.py collect_data --symbols AAPL --interval 1h
    python manage.py collect_data --all-intervals

Schedule examples:
    # Windows Task Scheduler: run every day at 6am
    # Linux cron: 0 6 * * 1-5 /path/to/venv/bin/python manage.py collect_data
"""

from __future__ import annotations

import logging
from django.core.management.base import BaseCommand, CommandError

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "SPY", "QQQ"]
DEFAULT_INTERVALS = ["1d"]
ALL_INTERVALS = ["1d", "1h", "15m", "5m"]


class Command(BaseCommand):
    help = "Collect and persist historical market data for configured symbols."

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            default=",".join(DEFAULT_SYMBOLS),
            help="Comma-separated list of ticker symbols (default: major US stocks).",
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1d",
            choices=["1d", "1h", "15m", "5m"],
            help="Data interval to collect (default: 1d).",
        )
        parser.add_argument(
            "--all-intervals",
            action="store_true",
            default=False,
            help="Collect all supported intervals (1d, 1h, 15m, 5m).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Print what would be collected without downloading.",
        )

    def handle(self, *args, **options):
        from myapp.data_pipeline import harmonize_market_data, persist_to_db

        symbols = [s.strip().upper() for s in options["symbols"].split(",") if s.strip()]
        intervals = ALL_INTERVALS if options["all_intervals"] else [options["interval"]]
        dry_run = options["dry_run"]

        self.stdout.write(self.style.SUCCESS(
            f"QuantixAI Data Collector — {len(symbols)} symbols × {len(intervals)} interval(s)"
        ))

        total_new = 0
        errors = 0

        for symbol in symbols:
            for interval in intervals:
                if dry_run:
                    self.stdout.write(f"  [DRY-RUN] Would collect {symbol} @ {interval}")
                    continue

                try:
                    self.stdout.write(f"  Collecting {symbol} @ {interval}...", ending=" ")
                    df = harmonize_market_data(symbol=symbol, interval=interval)
                    new_rows = persist_to_db(symbol=symbol, interval=interval, df=df)
                    total_new += new_rows
                    self.stdout.write(
                        self.style.SUCCESS(f"✓ {len(df)} rows, {new_rows} new")
                    )
                except Exception as exc:
                    self.stdout.write(self.style.ERROR(f"✗ Error: {exc}"))
                    logger.error("collect_data failed for %s @ %s: %s", symbol, interval, exc)
                    errors += 1

        if not dry_run:
            self.stdout.write(
                self.style.SUCCESS(
                    f"\nCollection complete: {total_new} new rows added. "
                    f"{errors} error(s)."
                )
            )
