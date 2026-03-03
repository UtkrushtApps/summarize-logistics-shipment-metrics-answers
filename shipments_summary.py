"""Logistics shipment metrics summarization.

This module provides functionality to read a shipments CSV file and
produce a tidy, per-carrier, per-date performance summary including:

- Average delivery time per day
- 7-day moving average of delivery time per carrier
- On-time delivery rate per day
- 7-day moving average of on-time rate per carrier

The code is designed to be used either as a library function or as a
standalone script that can output results as CSV or JSON. It uses
pandas and numpy for efficient data processing.

Example (CLI):
    python shipments_summary.py \
        --input shipments.csv \
        --output shipments_summary.csv \
        --format csv

Example (Python):
    from shipments_summary import summarize_shipments
    df_summary = summarize_shipments("shipments.csv")
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class ShipmentSummaryConfig:
    """Configuration options for shipments summarization.

    Attributes
    ----------
    input_path:
        Path to the input shipments CSV file.
    date_column_preference:
        Ordered list of column names to use as the per-shipment
        performance date (e.g., delivery date). The first matching
        column found in the CSV will be used.
    ship_date_candidates:
        Ordered list of possible ship date column names used to compute
        delivery time if a numeric ``delivery_time`` column is not
        present.
    delivery_time_column:
        Name of the column containing numeric delivery time, if it
        exists. If missing, the code will attempt to derive delivery
        time from ship and delivery dates.
    on_time_flag_candidates:
        Ordered list of possible boolean/flag columns that represent
        whether a shipment was delivered on time.
    promised_date_candidates:
        Ordered list of possible promised/expected delivery date
        columns used to derive on-time flags when explicit flags are
        not available.
    rolling_window_days:
        Size of the rolling window in days (as number of daily
        observations) used for moving averages.
    min_periods_rolling:
        Minimum number of observations in the window required to
        compute a rolling metric.
    """

    input_path: Path
    date_column_preference: Iterable[str] = (
        "delivery_date",
        "delivered_at",
        "delivered_date",
        "arrival_date",
        "date",  # generic fallback
    )
    ship_date_candidates: Iterable[str] = (
        "ship_date",
        "shipped_at",
        "pickup_date",
        "pickup_at",
    )
    delivery_time_column: str = "delivery_time"
    on_time_flag_candidates: Iterable[str] = (
        "delivered_on_time",
        "on_time",
        "on_time_flag",
    )
    promised_date_candidates: Iterable[str] = (
        "promised_delivery_date",
        "expected_delivery_date",
        "eta",
    )
    rolling_window_days: int = 7
    min_periods_rolling: int = 1


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _standardize_column_names(columns: Iterable[str]) -> list[str]:
    """Standardize column names to snake_case lowercase.

    This function trims whitespace, converts to lowercase, and replaces
    spaces and dashes with underscores.
    """

    cleaned = []
    for col in columns:
        col_clean = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        cleaned.append(col_clean)
    return cleaned


def _select_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df.

    Parameters
    ----------
    df:
        Input DataFrame.
    candidates:
        Iterable of candidate column names (assumed to be already
        standardized to match df.columns).

    Returns
    -------
    Optional[str]
        The first candidate that exists in df.columns, or ``None`` if
        none are present.
    """

    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coerce_to_datetime(series: pd.Series) -> pd.Series:
    """Safely convert a Series to pandas datetime, coercing errors to NaT."""

    return pd.to_datetime(series, errors="coerce", utc=False)


def _coerce_to_bool_flag(series: pd.Series) -> pd.Series:
    """Convert a Series of various representations into boolean (0/1) flags.

    The function handles:
    - existing boolean dtype
    - numeric 0/1 values
    - common textual values like "y", "yes", "true", "n", "no", "false"

    Any values that cannot be interpreted become NaN.
    """

    if series.dtype == bool:
        return series.astype(float)  # convert to float for easier aggregation

    # Try numeric first
    numeric = pd.to_numeric(series, errors="coerce")
    # If we have at least some non-NaN numeric values, assume 0/1 flags
    if numeric.notna().any():
        # Normalize to 0/1; treat any non-zero as 1
        numeric = (numeric != 0).astype(float)
        numeric.loc[series.isna()] = np.nan  # preserve NaNs
        return numeric

    # Fallback: interpret strings
    mapping = {
        "y": 1.0,
        "yes": 1.0,
        "true": 1.0,
        "t": 1.0,
        "n": 0.0,
        "no": 0.0,
        "false": 0.0,
        "f": 0.0,
    }

    def parse_value(x: object) -> float:
        if x is None:
            return np.nan
        s = str(x).strip().lower()
        return mapping.get(s, np.nan)

    return series.map(parse_value).astype(float)


# ---------------------------------------------------------------------------
# Core loading and transformation logic
# ---------------------------------------------------------------------------


def load_shipments(config: ShipmentSummaryConfig) -> pd.DataFrame:
    """Load and minimally clean the shipments CSV file.

    This function:

    - Reads the CSV from ``config.input_path``.
    - Standardizes column names to snake_case lowercase.
    - Attempts to parse all candidate date columns as datetimes.
    - Leaves delivery time and on-time flags to be derived later.

    Parameters
    ----------
    config:
        ShipmentSummaryConfig with file path and column preferences.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized columns and parsed dates.

    Raises
    ------
    FileNotFoundError
        If the input CSV file does not exist.
    ValueError
        If the CSV cannot be read or is empty.
    """

    input_path = Path(config.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV file does not exist: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to read CSV file '{input_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Input CSV file '{input_path}' is empty.")

    # Standardize column names
    df.columns = _standardize_column_names(df.columns)

    # Parse all obvious date-like columns; errors are coerced to NaT
    date_like_columns = set(config.date_column_preference) | set(config.ship_date_candidates) | set(
        config.promised_date_candidates
    )
    date_like_columns &= set(df.columns)

    for col in date_like_columns:
        df[col] = _coerce_to_datetime(df[col])

    return df


def _derive_per_shipment_metrics(df: pd.DataFrame, config: ShipmentSummaryConfig) -> pd.DataFrame:
    """Derive per-shipment performance fields (date, delivery_time, on_time).

    This function enriches the raw shipments DataFrame with the
    following columns:

    - ``performance_date``: the primary date for performance measurement,
      typically the delivery date, normalized to midnight.
    - ``delivery_time_days``: numeric delivery time in days.
    - ``delivered_on_time_flag``: float 0/1 indicator of on-time
      delivery, or NaN if unavailable.

    Rows missing critical information (carrier, performance_date,
    delivery_time) or with clearly invalid values (e.g., negative
    delivery time) are filtered out.
    """

    df = df.copy()

    # Ensure we have a carrier column
    if "carrier" not in df.columns:
        raise ValueError("Input data must contain a 'carrier' column.")

    # Normalize carrier representation
    df["carrier"] = df["carrier"].astype(str).str.strip()

    # Select performance date column
    perf_date_col = _select_first_existing_column(df, config.date_column_preference)
    if perf_date_col is None:
        raise ValueError(
            "Could not find a suitable delivery/performance date column. "
            f"Checked: {', '.join(config.date_column_preference)}"
        )

    df["performance_date"] = _coerce_to_datetime(df[perf_date_col]).dt.normalize()

    # Determine or derive delivery time in days
    if config.delivery_time_column in df.columns:
        delivery_time = pd.to_numeric(df[config.delivery_time_column], errors="coerce")
        df["delivery_time_days"] = delivery_time
    else:
        # Try to compute from ship_date and performance date (delivery)
        ship_col = _select_first_existing_column(df, config.ship_date_candidates)
        if ship_col is None:
            raise ValueError(
                "No explicit 'delivery_time' column and no usable ship date "
                "column found to derive it."
            )

        ship_dates = _coerce_to_datetime(df[ship_col])
        # delivery/performance date is already parsed above
        delivery_dates = df["performance_date"]

        delta = (delivery_dates - ship_dates).dt.total_seconds() / (24 * 3600)
        df["delivery_time_days"] = delta

    # Derive on-time flag if possible
    on_time_col = _select_first_existing_column(df, config.on_time_flag_candidates)

    if on_time_col is not None:
        df["delivered_on_time_flag"] = _coerce_to_bool_flag(df[on_time_col])
    else:
        # Attempt to derive from promised date vs actual delivery
        promised_col = _select_first_existing_column(df, config.promised_date_candidates)
        if promised_col is not None:
            promised_dates = _coerce_to_datetime(df[promised_col])
            delivered_on_time = (df["performance_date"] <= promised_dates) & promised_dates.notna()
            df["delivered_on_time_flag"] = delivered_on_time.astype(float)
        else:
            # No on-time information available
            df["delivered_on_time_flag"] = np.nan

    # Filter out rows with missing or invalid critical fields
    mask_valid = (
        df["carrier"].notna()
        & df["carrier"].astype(str).str.len().gt(0)
        & df["performance_date"].notna()
        & df["delivery_time_days"].notna()
    )

    # Remove negative or obviously unrealistic delivery times
    # (e.g., more than 365 days, which likely indicates bad data)
    mask_reasonable_time = (df["delivery_time_days"] >= 0) & (df["delivery_time_days"] <= 365)

    df_clean = df[mask_valid & mask_reasonable_time].copy()

    if df_clean.empty:
        raise ValueError("No valid shipment records remain after cleaning.")

    return df_clean


def compute_daily_carrier_summary(df: pd.DataFrame, config: ShipmentSummaryConfig) -> pd.DataFrame:
    """Aggregate per-shipment data into a daily per-carrier summary.

    Parameters
    ----------
    df:
        Raw shipments DataFrame with standardized column names.
    config:
        ShipmentSummaryConfig controlling behavior.

    Returns
    -------
    pandas.DataFrame
        Summary DataFrame with one row per (carrier, date) containing:

        - ``date``: performance date (datetime, normalized to midnight)
        - ``carrier``: carrier identifier
        - ``shipments``: number of shipments
        - ``avg_delivery_time_days``: average delivery time in days
        - ``avg_delivery_time_days_7d``: 7-day moving average of
          average delivery time per carrier
        - ``on_time_rate``: fraction of shipments delivered on time
        - ``on_time_rate_7d``: 7-day moving average of on-time rate
    """

    df_metrics = _derive_per_shipment_metrics(df, config)

    # Group by carrier and performance date
    group_cols = ["carrier", "performance_date"]

    grouped = df_metrics.groupby(group_cols, dropna=True)

    summary = grouped.agg(
        shipments=("delivery_time_days", "size"),
        avg_delivery_time_days=("delivery_time_days", "mean"),
        on_time_rate=("delivered_on_time_flag", "mean"),
    ).reset_index()

    # Clean up and enforce reasonable bounds on on_time_rate
    summary["on_time_rate"] = summary["on_time_rate"].clip(lower=0.0, upper=1.0)

    # Rename performance_date to date for output clarity
    summary = summary.rename(columns={"performance_date": "date"})

    # Sort before computing rolling metrics
    summary = summary.sort_values(["carrier", "date"]).reset_index(drop=True)

    # Compute 7-day rolling averages per carrier on sorted data
    window = config.rolling_window_days
    min_periods = config.min_periods_rolling

    summary["avg_delivery_time_days_7d"] = (
        summary.groupby("carrier")["avg_delivery_time_days"].transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).mean()
        )
    )

    if "on_time_rate" in summary.columns:
        summary["on_time_rate_7d"] = (
            summary.groupby("carrier")["on_time_rate"].transform(
                lambda s: s.rolling(window=window, min_periods=min_periods).mean()
            )
        )
    else:
        summary["on_time_rate_7d"] = np.nan

    # Re-order columns for readability
    ordered_cols = [
        "date",
        "carrier",
        "shipments",
        "avg_delivery_time_days",
        "avg_delivery_time_days_7d",
        "on_time_rate",
        "on_time_rate_7d",
    ]

    # Keep any existing columns in addition to ordered ones
    remaining_cols = [c for c in summary.columns if c not in ordered_cols]
    summary = summary[ordered_cols + remaining_cols]

    return summary


def summarize_shipments(input_path: str | Path) -> pd.DataFrame:
    """Public convenience function to produce the shipments summary.

    Parameters
    ----------
    input_path:
        Path to the input shipments CSV file.

    Returns
    -------
    pandas.DataFrame
        Tidy daily per-carrier performance summary.
    """

    config = ShipmentSummaryConfig(input_path=Path(input_path))
    raw_df = load_shipments(config)
    summary_df = compute_daily_carrier_summary(raw_df, config)
    return summary_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize logistics shipments into daily per-carrier metrics, "
            "including 7-day moving averages."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        required=True,
        help="Path to input shipments CSV file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        required=False,
        default=None,
        help=(
            "Path to write summary output. If omitted, results are written to "
            "stdout in the selected format."
        ),
    )
    parser.add_argument(
        "--format",
        "-f",
        dest="output_format",
        choices=["csv", "json"],
        default="csv",
        help="Output format: 'csv' (default) or 'json'.",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Logging level (default: WARNING).",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Command-line entry point.

    Returns an exit code suitable for ``sys.exit``.
    """

    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        summary_df = summarize_shipments(args.input_path)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Unexpected error while summarizing shipments: %s", exc)
        return 1

    # Output
    if args.output_format == "csv":
        if args.output_path:
            output_path = Path(args.output_path)
            try:
                summary_df.to_csv(output_path, index=False)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("Failed to write CSV to '%s': %s", output_path, exc)
                return 1
        else:
            # Write to stdout
            summary_df.to_csv(sys.stdout, index=False)
    else:  # json
        # Orient records is suitable for API responses
        json_str = summary_df.to_json(orient="records", date_format="iso")
        if args.output_path:
            output_path = Path(args.output_path)
            try:
                output_path.write_text(json_str, encoding="utf-8")
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.error("Failed to write JSON to '%s': %s", output_path, exc)
                return 1
        else:
            sys.stdout.write(json_str + "\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
