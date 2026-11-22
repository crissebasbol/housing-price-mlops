from datetime import date

import numpy as np
import pandas as pd

RANGES: dict[str, tuple[float, float]] = {
    "price": (10_000, 20_000_000),
    "house_size": (100, 30_000),
    "acre_lot": (0, 1_000),
    "bed": (1, 20),
    "bath": (1, 20),
}
FEATURE_RANGE_COLUMNS = ("house_size", "acre_lot", "bed", "bath")
REQUIRED_TEXT = ("status", "city", "state")
MIN_PREV_SOLD_YEAR = 1900
CLEAN_COLUMNS = (
    "price",
    "bed",
    "bath",
    "acre_lot",
    "house_size",
    "prev_sold_date",
    "status",
    "city",
    "state",
    "zip_code",
    "brokered_by",
)


def column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(np.nan, index=frame.index, dtype=object)


def none_for_missing(values: pd.Series) -> pd.Series:
    values = values.astype(object)
    return values.where(values.notna(), None)


def clean_text(values: pd.Series, lower: bool = False) -> pd.Series:
    text = values.astype("string").str.strip()
    if lower:
        text = text.str.lower()
    return none_for_missing(text.mask(text == ""))


def normalize_code(values: pd.Series) -> pd.Series:
    text = clean_text(values).astype("string")
    numeric = pd.to_numeric(values, errors="coerce")
    integer = numeric.notna() & (numeric == numeric.round())
    text[integer] = numeric[integer].astype("int64").astype(str)
    return none_for_missing(text)


def normalize_zip(values: pd.Series) -> pd.Series:
    text = clean_text(values).astype("string")
    numeric = pd.to_numeric(values, errors="coerce")
    valid_numeric = (
        numeric.notna() & (numeric >= 0) & (numeric < 100_000) & (numeric == numeric.round())
    )
    result = text.str.extract(r"^(\d{5})(?:-\d{4})?$", expand=False)
    result[valid_numeric] = numeric[valid_numeric].astype("int64").map("{:05d}".format)
    return none_for_missing(result)


def parse_dates(values: pd.Series, today: pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce", format="ISO8601")
    invalid = dates.notna() & ((dates.dt.year < MIN_PREV_SOLD_YEAR) | (dates > today))
    return dates.mask(invalid)


def normalize(raw: pd.DataFrame, ingestion_date: date | None = None) -> pd.DataFrame:
    today = pd.Timestamp(ingestion_date or date.today())
    clean = pd.DataFrame(index=raw.index)
    for name in RANGES:
        clean[name] = pd.to_numeric(column(raw, name), errors="coerce").astype(float)
    clean["prev_sold_date"] = parse_dates(column(raw, "prev_sold_date"), today)
    clean["status"] = clean_text(column(raw, "status"), lower=True)
    clean["city"] = clean_text(column(raw, "city"))
    clean["state"] = clean_text(column(raw, "state"))
    clean["zip_code"] = normalize_zip(column(raw, "zip_code"))
    clean["brokered_by"] = normalize_code(column(raw, "brokered_by"))
    return clean[list(CLEAN_COLUMNS)]


def quality_flags(clean: pd.DataFrame) -> pd.DataFrame:
    checks: dict[str, pd.Series] = {}
    for name, (low, high) in RANGES.items():
        values = clean[name]
        checks[f"{name}_missing"] = values.isna()
        checks[f"{name}_out_of_range"] = values.notna() & ((values < low) | (values > high))
    for name in REQUIRED_TEXT:
        checks[f"{name}_missing"] = clean[name].isna()
    checks["zip_code_invalid"] = clean["zip_code"].isna()
    return pd.DataFrame(checks, index=clean.index)


def rejection_reasons(flags: pd.DataFrame) -> pd.Series:
    rejected = flags[flags.any(axis=1)]
    return rejected.apply(lambda row: [name for name, hit in row.items() if hit], axis=1)


def iqr_outliers(values: pd.Series) -> int:
    values = values.dropna()
    if values.empty:
        return 0
    q1, q3 = values.quantile([0.25, 0.75])
    spread = q3 - q1
    return int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())


def quality_report(raw: pd.DataFrame, clean: pd.DataFrame, flags: pd.DataFrame) -> dict:
    rejected = flags.any(axis=1)
    valid = clean[~rejected]
    raw_dates = column(raw, "prev_sold_date")
    invalid_dates = int((raw_dates.notna() & clean["prev_sold_date"].isna()).sum())
    total = len(clean)
    return {
        "rows": total,
        "valid_rows": int((~rejected).sum()),
        "rejected_rows": int(rejected.sum()),
        "rejected_share": round(float(rejected.mean()), 4) if total else 0.0,
        "rejections_by_reason": {name: int(count) for name, count in flags.sum().items() if count},
        "nulls_by_column": {name: int(raw[name].isna().sum()) for name in raw.columns},
        "duplicate_rows": int(raw.duplicated().sum()),
        "invalid_prev_sold_date": invalid_dates,
        "iqr_outliers": {name: iqr_outliers(valid[name]) for name in RANGES},
        "price": {
            "min": float(valid["price"].min()) if len(valid) else None,
            "median": float(valid["price"].median()) if len(valid) else None,
            "max": float(valid["price"].max()) if len(valid) else None,
        },
    }
