from datetime import date

import pandas as pd

from housing_ml.cleaning import (
    normalize,
    normalize_code,
    normalize_zip,
    quality_flags,
    quality_report,
    rejection_reasons,
)


def test_zip_codes_keep_leading_zeros():
    zips = normalize_zip(pd.Series([6016.0, "02134", "98101-1234", None, -5, "abc"]))
    assert zips.tolist() == ["06016", "02134", "98101", None, None, None]


def test_codes_drop_float_suffix():
    codes = normalize_code(pd.Series([101640.0, "agency", None]))
    assert codes.tolist() == ["101640", "agency", None]


def test_normalize_types_and_dates(raw):
    rows = raw.head(3).copy()
    rows.loc[0, "prev_sold_date"] = "2099-01-01"
    rows.loc[1, "prev_sold_date"] = "not a date"
    rows.loc[2, "status"] = "  FOR_SALE "
    clean = normalize(rows, date(2026, 11, 21))
    assert clean["prev_sold_date"].isna().tolist()[:2] == [True, True]
    assert clean.loc[2, "status"] == "for_sale"
    assert clean["zip_code"].str.len().eq(5).all()


def test_quality_rejects_out_of_range_rows(raw):
    rows = raw.head(4).copy()
    rows.loc[0, "price"] = 1
    rows.loc[1, "bed"] = 444
    rows.loc[2, "state"] = "  "
    clean = normalize(rows)
    flags = quality_flags(clean)
    reasons = rejection_reasons(flags)
    assert reasons.loc[0] == ["price_out_of_range"]
    assert reasons.loc[1] == ["bed_out_of_range"]
    assert reasons.loc[2] == ["state_missing"]
    assert 3 not in reasons.index

    report = quality_report(rows, clean, flags)
    assert report["rejected_rows"] == 3
    assert report["rejections_by_reason"]["price_out_of_range"] == 1
