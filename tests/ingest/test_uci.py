from pathlib import Path

import pandas as pd
import pytest

from saveproof.contract import validate
from saveproof.ingest.uci import load_uci_steel

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_uci_midnight_row_is_end_of_its_labelled_day():
    """Real file order: "01/01 00:00" follows "01/01 23:45" (NSM = 0) and ends Jan 1."""
    daily = load_uci_steel(FIXTURES / "uci_steel_midnight.csv")
    validate(daily)

    assert list(daily["period_start"]) == list(pd.to_datetime(["2018-01-01", "2018-01-02"]))
    assert daily.loc[0, "kwh"] == pytest.approx(6.0)
    assert daily.loc[0, "production"] == pytest.approx(60.0)
    assert daily.loc[0, "coverage_pct"] == pytest.approx(3 / 96 * 100)
    assert daily.loc[1, "kwh"] == pytest.approx(4.0)
    assert daily.loc[1, "production"] == pytest.approx(40.0)
    assert daily.loc[1, "coverage_pct"] == pytest.approx(1 / 96 * 100)


def test_uci_contract_metadata_and_production_source_are_exact():
    daily = load_uci_steel(FIXTURES / "uci_steel_midnight.csv")

    assert daily["production_unit"].tolist() == ["proxy:kvarh_lagging"] * 2
    assert daily["source"].tolist() == ["real_meter"] * 2
    assert daily["dataset"].tolist() == ["uci_steel"] * 2
    assert daily["operating_days"].tolist() == [1.0, 1.0]
    assert daily["event"].isna().all()
    assert daily["production"].tolist() == [60.0, 40.0]


def test_uci_rejects_midnight_row_that_breaks_time_order(tmp_path):
    lines = (FIXTURES / "uci_steel_midnight.csv").read_text(encoding="utf-8-sig").splitlines()
    # "02/01 00:00" before "02/01 00:15" would be the start of Jan 2, not the end: not our file.
    lines[3] = lines[3].replace("01/01/2018 00:00", "02/01/2018 00:00")
    path = tmp_path / "uci.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="column 'date': row 3"):
        load_uci_steel(path)
