from dataclasses import asdict, dataclass, field

import pandas as pd

NUMERIC_COLUMNS = (
    "brokered_by",
    "price",
    "bed",
    "bath",
    "acre_lot",
    "street",
    "zip_code",
    "house_size",
)
TEXT_COLUMNS = ("status", "city", "state", "prev_sold_date")
EXPECTED_COLUMNS = NUMERIC_COLUMNS + TEXT_COLUMNS
REQUIRED_COLUMNS = (
    "price",
    "bed",
    "bath",
    "acre_lot",
    "house_size",
    "status",
    "city",
    "state",
    "zip_code",
)
MAX_UNCONVERTIBLE_SHARE = 0.05


@dataclass
class SchemaReport:
    status: str
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    unconvertible: dict[str, float] = field(default_factory=dict)
    received: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def received_schema(frame: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in frame.dtypes.items()}


def validate_schema(frame: pd.DataFrame) -> SchemaReport:
    columns = set(frame.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in columns]
    extra = sorted(columns - set(EXPECTED_COLUMNS))
    unconvertible: dict[str, float] = {}
    coerced = False
    for column in NUMERIC_COLUMNS:
        if column not in columns or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        coerced = True
        values = frame[column]
        converted = pd.to_numeric(values, errors="coerce")
        share = float((converted.isna() & values.notna()).mean()) if len(values) else 0.0
        if share:
            unconvertible[column] = round(share, 4)

    broken = [
        c
        for c, share in unconvertible.items()
        if c in REQUIRED_COLUMNS and share > MAX_UNCONVERTIBLE_SHARE
    ]
    if any(c in REQUIRED_COLUMNS for c in missing) or broken:
        status = "invalid"
    elif missing or extra or coerced:
        status = "adapted"
    else:
        status = "ok"
    return SchemaReport(status, missing, extra, unconvertible, received_schema(frame))
