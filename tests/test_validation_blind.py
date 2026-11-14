"""Blind-test scoring (docs/validation_plan.md §2 wrong side, §6 metrics and P1-P5) on hand-made
tables: every expected number is worked out in the test."""

import numpy as np
import pandas as pd
import pytest

from saveproof.validation import (
    blind_metrics,
    blind_metrics_table,
    blind_scoring_table,
    p1,
    p2,
    p3,
    p4,
    p5,
)

NAN = float("nan")


def plants(method, stratum, cases):
    """cases: (true_pct, estimate_pct, verdict, band_low_pct, band_high_pct) per plant."""
    return pd.DataFrame(
        [
            {
                "plant_id": f"{method}_{stratum}_{i}",
                "method": method,
                "stratum": stratum,
                "granularity": "daily",
                "fixed_share_band": "<0.2",
                "true_savings_pct": true,
                "savings_pct": est,
                "verdict": verdict,
                "band_low_pct": low,
                "band_high_pct": high,
            }
            for i, (true, est, verdict, low, high) in enumerate(cases)
        ]
    )


def table(*frames):
    return pd.concat(frames, ignore_index=True)


# ---------- metrics ----------


def test_blind_metrics_hand_example():
    """Errors over estimated plants: 7-5=2, 8-15=-7, 12.5-12=0.5, 3.5-3=0.5 -> MAE 10/4 = 2.5,
    bias -4/4 = -1.0. Wrong side: plant 0 (verified, true 5) and plant 1 (below, true 15) -> 2/5.
    Abstained: plants 2 (insufficient data) and 3 -> 2/5. True < 10: plants 0, 2, 4; one verified
    -> 1/3. Decided (verified or below): plants 0, 1, 4; only plant 4's band holds the truth -> 1/3.
    """
    rows = plants(
        "saveproof",
        "S1",
        [
            (5.0, 7.0, "verified", 10.2, 13.0),
            (15.0, 8.0, "below", 6.0, 9.9),
            (8.0, NAN, "insufficient_data", NAN, NAN),
            (12.0, 12.5, "likely_not_proven", 9.0, 16.0),
            (3.0, 3.5, "below", 1.0, 6.0),
        ],
    )
    m = blind_metrics(rows)
    assert m["n"] == 5 and m["n_estimated"] == 4
    assert m["mae_pp"] == pytest.approx(2.5)
    assert m["bias_pp"] == pytest.approx(-1.0)
    assert m["wrong_side_rate"] == pytest.approx(0.4)
    assert m["abstention_rate"] == pytest.approx(0.4)
    assert m["n_true_below_10"] == 3 and m["n_false_verified"] == 1
    assert m["false_verified_rate"] == pytest.approx(1 / 3)
    assert m["n_decided"] == 3
    assert m["band_coverage"] == pytest.approx(1 / 3)


def test_true_saving_of_exactly_10_counts_as_at_or_above():
    rows = plants("naive", "S1", [(10.0, 9.0, "below", NAN, NAN)])
    assert blind_metrics(rows)["wrong_side_rate"] == 1.0
    assert blind_metrics(rows)["n_true_below_10"] == 0


def test_comparators_have_no_band_coverage():
    rows = plants("bee_sec", "S1", [(5.0, 12.0, "verified", NAN, NAN)])
    assert blind_metrics(rows)["band_coverage"] is None


def test_metrics_table_has_every_method_and_breakdown():
    long = table(
        plants("saveproof", "S1", [(5.0, 5.0, "below", 3.0, 7.0)]),
        plants("naive", "S1", [(5.0, 6.0, "below", NAN, NAN)]),
        plants("saveproof", "S2", [(15.0, 14.0, "verified", 11.0, 17.0)]),
        plants("naive", "S2", [(15.0, 30.0, "verified", NAN, NAN)]),
    )
    out = blind_metrics_table(long)
    keys = set(zip(out["method"], out["group"], out["value"], strict=True))
    assert ("saveproof", "overall", "all") in keys
    assert ("naive", "stratum", "S2") in keys
    assert ("saveproof", "granularity", "daily") in keys
    assert ("naive", "fixed_share_band", "<0.2") in keys
    row = out[(out["method"] == "naive") & (out["group"] == "overall")].iloc[0]
    assert row["mae_pp"] == pytest.approx(8.0)  # (1 + 15) / 2


# ---------- joining verdicts and truth ----------


def verdicts_and_truth():
    verdicts = pd.DataFrame(
        {
            "plant_id": ["plant_000", "plant_001"],
            "status": ["ok", "insufficient_data"],
            "savings_pct": [12.0, NAN],
            "verdict": ["verified", None],
            "band_low_pct": [10.5, NAN],
            "band_high_pct": [13.5, NAN],
            "naive_savings_pct": [3.0, 25.0],
            "naive_verdict": ["below", "verified"],
            "bee_sec_savings_pct": [9.0, 22.0],
            "bee_sec_verdict": ["below", "verified"],
        }
    )
    truth = pd.DataFrame(
        {
            "plant_id": ["plant_001", "plant_000"],
            "true_savings_pct": [4.0, 11.0],
            "stratum": ["S2", "S1"],
            "granularity": ["monthly", "daily"],
            "sector": ["foundry", "dairy_cold_chain"],
            "fixed_share": [0.2, 0.35],
            "production_change_pct": [20.0, 1.0],
        }
    )
    return verdicts, truth


def test_scoring_table_is_one_row_per_plant_and_method():
    long = blind_scoring_table(*verdicts_and_truth())
    assert len(long) == 6
    ours = long[long["method"] == "saveproof"].set_index("plant_id")
    assert ours.at["plant_000", "true_savings_pct"] == 11.0
    assert ours.at["plant_001", "verdict"] == "insufficient_data"
    assert np.isnan(ours.at["plant_001", "savings_pct"])
    naive = long[long["method"] == "naive"].set_index("plant_id")
    assert naive.at["plant_001", "savings_pct"] == 25.0
    assert np.isnan(naive.at["plant_001", "band_low_pct"])
    bands = long.drop_duplicates("plant_id").set_index("plant_id")["fixed_share_band"]
    assert bands.to_dict() == {"plant_000": ">0.3", "plant_001": "0.2-0.3"}


@pytest.mark.parametrize("share, band", [(0.1, "<0.2"), (0.2, "0.2-0.3"), (0.3, "0.2-0.3")])
def test_fixed_share_band_edges(share, band):
    verdicts, truth = verdicts_and_truth()
    truth["fixed_share"] = share
    assert set(blind_scoring_table(verdicts, truth)["fixed_share_band"]) == {band}


def test_scoring_table_raises_when_plants_differ():
    verdicts, truth = verdicts_and_truth()
    with pytest.raises(ValueError, match="plant_001"):
        blind_scoring_table(verdicts, truth.iloc[[1]])
    with pytest.raises(ValueError, match="plant_000"):
        blind_scoring_table(verdicts.iloc[[1]], truth)


# ---------- P1-P5 pass/fail tables ----------
# Plants with true 15: "verified" is right, "below" is wrong. True 5: the reverse.
RIGHT = (15.0, 15.0, "verified", 12.0, 18.0)
WRONG = (15.0, 5.0, "below", 3.0, 7.0)
ABSTAIN = (15.0, 11.0, "likely_not_proven", 8.0, 14.0)


def test_p1_passes_when_ours_is_strictly_lower_in_core_and_swing_strata():
    long = table(
        plants("saveproof", "S2", [RIGHT, RIGHT, ABSTAIN, RIGHT]),  # 0/4
        plants("naive", "S2", [RIGHT, WRONG, RIGHT, RIGHT]),  # 1/4
        plants("bee_sec", "S2", [WRONG, WRONG, RIGHT, RIGHT]),  # 2/4
        plants("saveproof", "S4", [WRONG] * 4),  # S4 is not part of P1
        plants("naive", "S4", [RIGHT] * 4),
        plants("bee_sec", "S4", [RIGHT] * 4),
    )
    result = p1(long)
    assert result["pass"] is True
    assert result["S1-S3"]["saveproof"] == 0.0
    assert result["S2+S3"]["bee_sec"] == 0.5


def test_p1_fails_on_a_tie_in_s2_s3():
    long = table(
        plants("saveproof", "S1", [RIGHT, RIGHT]),
        plants("naive", "S1", [WRONG, WRONG]),
        plants("bee_sec", "S1", [WRONG, WRONG]),
        plants("saveproof", "S3", [WRONG, RIGHT]),
        plants("naive", "S3", [WRONG, RIGHT]),  # tie: not strictly lower
        plants("bee_sec", "S3", [WRONG, WRONG]),
    )
    result = p1(long)
    assert result["S1-S3"]["saveproof"] < result["S1-S3"]["naive"]
    assert result["pass"] is False


FALSE_VERIFIED = (5.0, 12.0, "verified", 10.5, 13.5)
TRUE_BELOW = (5.0, 5.0, "below", 3.0, 7.0)


@pytest.mark.parametrize(
    "core_false, s4_false, expected",
    [(1, 0, True), (2, 0, False), (0, 2, False), (1, 1, True)],
)
def test_p2_false_verified_at_most_5pct_in_core_and_in_s4(core_false, s4_false, expected):
    """Core: 20 plants with true < 10, so 1 false = 5% (passes, <=), 2 = 10% fails. S4: 20 too."""
    long = table(
        plants("saveproof", "S1", [FALSE_VERIFIED] * core_false + [TRUE_BELOW] * (20 - core_false)),
        plants("saveproof", "S4", [FALSE_VERIFIED] * s4_false + [TRUE_BELOW] * (20 - s4_false)),
        plants("naive", "S1", [FALSE_VERIFIED] * 20),  # comparators are not judged by P2
    )
    assert p2(long)["pass"] is expected


def covered(n):
    return [(15.0, 15.0, "verified", 12.0, 18.0)] * n


def missed(n):
    return [(15.0, 25.0, "verified", 22.0, 28.0)] * n


@pytest.mark.parametrize(
    "n_covered, n_missed, expected",
    [(9, 1, True), (17, 3, True), (8, 2, False), (10, 0, False), (97, 3, True), (98, 2, False)],
)
def test_p3_coverage_must_lie_within_085_and_097(n_covered, n_missed, expected):
    long = table(
        plants("saveproof", "S1", covered(n_covered) + missed(n_missed)),
        plants("saveproof", "S1", [ABSTAIN] * 5),  # abstentions are not scored
        plants("saveproof", "S4", missed(10)),  # S4 is not part of P3
    )
    result = p3(long)
    assert result["n_decided"] == n_covered + n_missed
    assert result["pass"] is expected


def test_p3_is_none_without_decided_plants():
    assert p3(plants("saveproof", "S1", [ABSTAIN] * 3))["pass"] is None


def est(true, estimate):
    verdict = "verified" if estimate >= 10 else "below"
    return (true, estimate, verdict, NAN, NAN)


@pytest.mark.parametrize(
    "ours_s1_err, expected",
    [(2.0, True), (2.0 + 1.0, True), (2.0 + 1.5, False)],
)
def test_p4_beats_sec_in_s2_s3_and_at_most_1pp_worse_in_s1(ours_s1_err, expected):
    long = table(
        plants("saveproof", "S2", [est(10, 11), est(10, 9)]),  # MAE 1
        plants("bee_sec", "S2", [est(10, 14), est(10, 6)]),  # MAE 4
        plants("saveproof", "S1", [est(10, 10 + ours_s1_err)]),
        plants("bee_sec", "S1", [est(10, 12)]),  # MAE 2
    )
    assert p4(long)["pass"] is expected


def test_p4_fails_when_sec_wins_in_s2_s3():
    long = table(
        plants("saveproof", "S3", [est(10, 14)]),
        plants("bee_sec", "S3", [est(10, 12)]),
        plants("saveproof", "S1", [est(10, 10)]),
        plants("bee_sec", "S1", [est(10, 10)]),
    )
    assert p4(long)["pass"] is False


@pytest.mark.parametrize("errors, expected", [((0.5, 0.5), True), ((-1.2, -1.2), False),
                                               ((3.0, -3.0), True)])  # fmt: skip
def test_p5_abs_bias_below_1pp_excluding_s4(errors, expected):
    long = table(
        plants("saveproof", "S1", [est(10, 10 + e) for e in errors]),
        plants("saveproof", "S4", [est(10, 20)] * 5),  # S4 bias ignored
    )
    assert p5(long)["pass"] is expected
