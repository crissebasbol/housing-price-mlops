from housing_ml.schema import validate_schema


def test_complete_batch_is_ok(raw):
    report = validate_schema(raw)
    assert report.status == "ok"
    assert report.missing == [] and report.extra == []


def test_extra_column_is_adapted(raw):
    report = validate_schema(raw.assign(garage=1))
    assert report.status == "adapted"
    assert report.extra == ["garage"]


def test_missing_optional_column_is_adapted(raw):
    report = validate_schema(raw.drop(columns=["street"]))
    assert report.status == "adapted"
    assert report.missing == ["street"]


def test_numbers_as_text_are_adapted(raw):
    report = validate_schema(raw.assign(price=raw["price"].astype(str)))
    assert report.status == "adapted"
    assert report.unconvertible == {}


def test_missing_required_column_is_invalid(raw):
    assert validate_schema(raw.drop(columns=["price"])).status == "invalid"


def test_unconvertible_required_column_is_invalid(raw):
    assert validate_schema(raw.assign(bed="many")).status == "invalid"
