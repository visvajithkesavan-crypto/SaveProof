import csv
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from saveproof.contract import validate
from saveproof.ingest.irish import load_irish

FIXTURES = Path(__file__).parents[1] / "fixtures"


def make_irish_dataset(root: Path) -> Path:
    extracted = root / "extracted"
    metadata = root / "metadata"
    extracted.mkdir(parents=True)
    metadata.mkdir()
    shutil.copy(FIXTURES / "meter_data_quality_log.csv", metadata)

    rows = pd.read_csv(FIXTURES / "irish_initial_partial.csv").to_dict("records")
    segments = pd.read_csv(FIXTURES / "irish_dst_segments.csv")
    for segment in segments.to_dict("records"):
        start = segment["window_start_utc"]
        count = int(segment["interval_count"])
        mi_a = float(segment["mi_a"])
        mi_b = None if pd.isna(segment["mi_b"]) else float(segment["mi_b"])
        interval_start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        for index in range(count):
            timestamp = interval_start + timedelta(minutes=15 * index)
            rows.append(
                {
                    "window_start_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "mi_a": mi_a,
                    "mi_b": mi_b,
                }
            )

    meter_path = extracted / "site_15min_ALLphase.csv"
    with meter_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["window_start_utc", "mi_a", "mi_b"])
        writer.writeheader()
        writer.writerows(rows)
    return meter_path


def test_irish_local_days_use_dublin_dates_dst_coverage_and_hierarchical_meters(tmp_path):
    meter_path = make_irish_dataset(tmp_path / "irish")
    daily = load_irish(meter_path)
    validate(daily)

    dates = daily["period_start"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == [
        "2025-03-29",
        "2025-03-30",
        "2025-05-23",
        "2025-05-24",
        "2025-10-26",
    ]
    assert "2025-03-28" not in dates
    coverage = dict(zip(dates, daily["coverage_pct"], strict=True))
    assert coverage["2025-03-29"] == pytest.approx(100.0)
    assert coverage["2025-03-30"] == pytest.approx(91 / 92 * 100)
    assert coverage["2025-05-23"] == pytest.approx(100.0)
    assert coverage["2025-05-24"] == pytest.approx(95 / 96 * 100)
    assert coverage["2025-10-26"] == pytest.approx(99 / 100 * 100)

    may_23 = daily.loc[daily["period_start"] == pd.Timestamp("2025-05-23")].iloc[0]
    assert may_23["kwh"] == pytest.approx(960.0)
    assert may_23["production"] == pytest.approx(192.0)
    assert may_23["event"] == "meter_fault"

    may_24 = daily.loc[daily["period_start"] == pd.Timestamp("2025-05-24")].iloc[0]
    assert np.isnan(may_24["production"])
    assert may_24["event"] is None
    assert daily["production_unit"].eq("proxy:pv_kwh").all()
    assert daily["source"].eq("real_meter").all()
    assert daily["dataset"].eq("irish_mi_a").all()
    assert daily["operating_days"].eq(1.0).all()
