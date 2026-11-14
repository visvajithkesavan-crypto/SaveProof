"""Generate a simulated plant set for the blind test (docs/validation_plan.md §6). Label: simulated.

    .\\.venv\\Scripts\\python.exe -m experiments.make_plant_set --name dev --seed 1 --n 50
    .\\.venv\\Scripts\\python.exe -m experiments.make_plant_set --name blind --seed <secret> --n 250

Writes data/processed/<name>/plant_NNN.csv and plant_NNN_meta.json (contract table and PlantMeta
only: no truth, no SimParams) and the truth, separately, to data/<name>_truth.csv. Strata, then
granularity, then sector cycle through the plants and the order is shuffled, so a plant number
reveals nothing; shares are exact when n is a multiple of 16, otherwise as even as possible. The
seed is never printed or written anywhere. Refuses to overwrite an existing set.
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from experiments.common import REPO, write_plant
from saveproof.ingest.simulator import GRANULARITIES, STRATA, sample_params, simulate_plant
from saveproof.validation import TRUTH_COLUMNS

DATA_DIR = REPO / "data"
SECTORS = ("foundry", "dairy_cold_chain")


def composition(n: int, rng: np.random.Generator) -> list[tuple[str, str, str]]:
    """(stratum, granularity, sector) per plant, in shuffled order."""
    cells = [(STRATA[i % 4], GRANULARITIES[(i // 4) % 2], SECTORS[(i // 8) % 2]) for i in range(n)]
    return [cells[i] for i in rng.permutation(n)]


def make_plant_set(name: str, seed: int, n: int) -> pd.DataFrame:
    """Simulate and write n plants; returns the truth table (also written to data/)."""
    out_dir = DATA_DIR / "processed" / name
    truth_path = DATA_DIR / f"{name}_truth.csv"
    if truth_path.exists() or (out_dir.exists() and any(out_dir.iterdir())):
        raise FileExistsError(f"plant set {name!r} already exists in {DATA_DIR}; delete it first")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    truth_rows = []
    for i, (stratum, granularity, sector) in enumerate(composition(n, rng)):
        params = sample_params(stratum, granularity, sector, rng)
        df, meta, truth = simulate_plant(params, rng, plant_id=f"plant_{i:03d}")
        write_plant(out_dir, df, meta)
        truth_rows.append(
            {
                "plant_id": meta.plant_id,
                "true_savings_pct": truth.true_savings_pct,
                "stratum": stratum,
                "granularity": granularity,
                "sector": sector,
                "fixed_share": params.fixed_share,
                "production_change_pct": params.production_change_pct,
            }
        )
    truth_table = pd.DataFrame(truth_rows, columns=list(TRUTH_COLUMNS))
    truth_table.to_csv(truth_path, index=False)
    return truth_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    args = parser.parse_args(argv)
    truth = make_plant_set(args.name, args.seed, args.n)
    print(f"wrote {len(truth)} simulated plants to {DATA_DIR / 'processed' / args.name}")
    for column in ("stratum", "granularity", "sector"):
        print(f"{column}: {dict(sorted(Counter(truth[column]).items()))}")


if __name__ == "__main__":
    main()
