# Solution Steps

1. Create a new Python file named `shipments_summary.py` and add a module-level docstring explaining that the script reads a shipments CSV and produces a daily, per-carrier summary with delivery-time and on-time metrics.

2. Import the required standard-library modules: `argparse`, `logging`, `sys`, `dataclasses.dataclass`, `pathlib.Path`, and typing helpers like `Iterable` and `Optional`. Then import `numpy as np` and `pandas as pd`.

3. Define a `ShipmentSummaryConfig` dataclass that holds configuration parameters: the input CSV path, preferred date columns for performance date, candidate ship date columns, the delivery time column name, candidate on-time flag columns, candidate promised-date columns, and rolling window settings (window size and min_periods). Provide sensible default tuples for the column preferences and a default rolling window size of 7 days.

4. Implement a helper function `_standardize_column_names(columns)` that takes an iterable of column names, strips whitespace, converts to lowercase, and replaces spaces and dashes with underscores. Return the cleaned list and later assign it to `df.columns` after reading the CSV.

5. Implement `_select_first_existing_column(df, candidates)` which loops over the iterable of candidate names and returns the first that exists in `df.columns`, or `None` if none are found. This will be used to flexibly select date and flag columns from differently-named schemas.

6. Implement `_coerce_to_datetime(series)` that calls `pd.to_datetime(series, errors="coerce", utc=False)` so any unparsable values become `NaT` instead of raising errors.

7. Implement `_coerce_to_bool_flag(series)` to standardize on-time flags: handle boolean dtype by casting to float, attempt numeric coercion and treat non-zero as 1.0 and zero as 0.0, and if that fails, map common textual values ("y", "yes", "true", "n", "no", "false", etc.) to 1.0/0.0, returning a float Series with NaN for unknown values.

8. Implement `load_shipments(config)` which verifies that the CSV file exists, reads it with `pd.read_csv`, raises a `ValueError` if read fails or the DataFrame is empty, standardizes the column names using `_standardize_column_names`, then identifies all date-like columns from the union of the preferred performance-date, ship-date, and promised-date candidate sets that actually exist in the DataFrame and converts them to datetime using `_coerce_to_datetime`. Return this minimally cleaned DataFrame.

9. Implement `_derive_per_shipment_metrics(df, config)` that operates on a copy of the DataFrame: ensure a `carrier` column exists, normalize `carrier` as stripped strings, select the performance-date column using `_select_first_existing_column` on `config.date_column_preference`, convert it to datetime and normalize to midnight into a new `performance_date` column, then derive `delivery_time_days` either from an existing numeric `config.delivery_time_column` or by subtracting a selected ship-date column from `performance_date` and converting the timedelta to days.

10. Within `_derive_per_shipment_metrics`, derive an on-time flag: if any of the `on_time_flag_candidates` is present, convert it with `_coerce_to_bool_flag` into `delivered_on_time_flag`. Otherwise, if a promised-date column exists, compute `delivered_on_time_flag` as 1.0 when `performance_date <= promised_date` and promised date is not null, else 0.0. If neither flags nor promised dates exist, set `delivered_on_time_flag` to NaN for all rows.

11. Still in `_derive_per_shipment_metrics`, build a validity mask that keeps only rows with non-empty `carrier`, non-null `performance_date`, and non-null `delivery_time_days`. Additionally, build a reasonableness mask that keeps only rows where `delivery_time_days` is between 0 and 365 (inclusive). Filter the DataFrame on both masks; if the result is empty, raise a `ValueError`. Return this cleaned, enriched DataFrame.

12. Implement `compute_daily_carrier_summary(df, config)` that first calls `_derive_per_shipment_metrics` to get a per-shipment metrics DataFrame, then groups it by `['carrier', 'performance_date']` and aggregates: `shipments` as size, `avg_delivery_time_days` as mean of `delivery_time_days`, and `on_time_rate` as mean of `delivered_on_time_flag`. Reset the index and clip `on_time_rate` to [0.0, 1.0].

13. In `compute_daily_carrier_summary`, rename `performance_date` to `date`, then sort the summary by `['carrier', 'date']`. Use `groupby('carrier')['avg_delivery_time_days'].transform(lambda s: s.rolling(window=config.rolling_window_days, min_periods=config.min_periods_rolling).mean())` to compute a per-carrier 7-day rolling average `avg_delivery_time_days_7d`. Similarly, compute `on_time_rate_7d` as a rolling mean of `on_time_rate` grouped by `carrier`.

14. Reorder the summary DataFrame columns in `compute_daily_carrier_summary` to a logical order: `date`, `carrier`, `shipments`, `avg_delivery_time_days`, `avg_delivery_time_days_7d`, `on_time_rate`, `on_time_rate_7d`, followed by any remaining columns, and return this tidy summary DataFrame.

15. Implement a convenience function `summarize_shipments(input_path)` that constructs a `ShipmentSummaryConfig` from the input path, calls `load_shipments(config)`, then `compute_daily_carrier_summary(raw_df, config)`, and returns the resulting summary DataFrame. This is the function that an API endpoint could call.

16. Add CLI utilities: implement `_parse_args(argv)` using `argparse` to accept `--input/-i` (required input CSV path), optional `--output/-o` (output file path, default stdout), `--format/-f` (`csv` or `json`, default `csv`), and `--log-level` (default `WARNING`). Implement `_configure_logging(level)` to set up basic logging with the chosen level.

17. Implement `main(argv=None)` that parses arguments, configures logging, calls `summarize_shipments` inside a `try` block, and handles `FileNotFoundError` and `ValueError` by logging an error and returning exit code 1. For unexpected exceptions, log a stack trace and return 1. On success, write the summary as CSV (to file or stdout) or JSON (records-orient, ISO dates) based on `--format` and `--output`, and return 0.

18. At the bottom of `shipments_summary.py`, add the standard `if __name__ == "__main__":` guard that calls `raise SystemExit(main())`, so the script can be executed directly from the command line while still being importable as a module.

