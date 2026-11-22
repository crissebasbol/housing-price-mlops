import numpy as np
import pandas as pd
import pytest

STATES = {"Texas": 1.0, "Florida": 1.2, "California": 2.5, "Ohio": 0.8, "New York": 1.8}


def make_records(rows: int = 2_000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    states = rng.choice(list(STATES), rows)
    house_size = rng.uniform(600, 4_000, rows).round()
    bed = rng.integers(1, 6, rows)
    bath = rng.integers(1, 4, rows)
    price = (
        house_size
        * 150
        * np.array([STATES[s] for s in states])
        * (1 + 0.05 * bath)
        * rng.lognormal(0, 0.1, rows)
    ).round(-2)
    return [
        {
            "brokered_by": float(rng.integers(1, 300)),
            "status": str(rng.choice(["for_sale", "sold"])),
            "price": float(price[i]),
            "bed": float(bed[i]),
            "bath": float(bath[i]),
            "acre_lot": round(float(rng.uniform(0.05, 2)), 2),
            "street": float(rng.integers(1, 10_000_000)),
            "city": f"City {rng.integers(1, 30)}",
            "state": str(states[i]),
            "zip_code": float(rng.integers(1_000, 99_999)),
            "house_size": float(house_size[i]),
            "prev_sold_date": f"{rng.integers(1995, 2022)}-{rng.integers(1, 13):02d}-15",
        }
        for i in range(rows)
    ]


@pytest.fixture
def records() -> list[dict]:
    return make_records()


@pytest.fixture
def raw(records) -> pd.DataFrame:
    return pd.DataFrame(records)
