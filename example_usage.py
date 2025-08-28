#!/usr/bin/env python3
"""
ADIA Refactored Package - Example Usage (minimal, no plotting)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure we can import the package when running this file directly
THIS_DIR = Path(__file__).parent
PARENT = THIS_DIR.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from adia_refactored import (
    compute_predictors_for_values,
    run_batch,
    validate_config,
)


def single_series_demo() -> None:
    np.random.seed(42)
    n = 120
    break_point = 60
    values = np.concatenate([
        np.random.normal(0, 1.0, break_point),
        np.random.normal(1.5, 1.5, n - break_point),
    ])
    periods = np.concatenate([np.zeros(break_point), np.ones(n - break_point)])

    preds, meta = compute_predictors_for_values(values, periods, B_boot=20, energy_enable=False)

    print("Single-series demo:")
    print(f"  n={n}, break_point={break_point}")
    print(f"  p_mean={preds['p_mean']:.4f}, p_var={preds['p_var']:.4f}, p_MWU={preds['p_MWU']:.4f}")
    print(f"  p_mu_lag1={preds['p_mu_lag1']:.4f}, p_sigma_lag1={preds['p_sigma_lag1']:.4f}")


def batch_demo(tmp_path: Path) -> None:
    # Create 4 simple synthetic series and write to parquet expected schema
    frames = []
    for i in range(4):
        np.random.seed(100 + i)
        n = 80
        bp = 40
        if i % 2 == 0:
            vals = np.concatenate([
                np.random.normal(0, 1.0, bp),
                np.random.normal(1.2, 1.0, n - bp),
            ])
        else:
            vals = np.concatenate([
                np.random.normal(0, 0.8, bp),
                np.random.normal(0, 1.8, n - bp),
            ])
        periods = np.concatenate([np.zeros(bp), np.ones(n - bp)])
        df = pd.DataFrame({"value": vals, "period": periods})
        df.index = pd.MultiIndex.from_tuples([(f"series_{i}", t) for t in range(n)], names=["id", "time"])
        frames.append(df)
    all_df = pd.concat(frames).sort_index()

    in_path = tmp_path / "sample_batch.parquet"
    out_pred = tmp_path / "sample_predictors.parquet"
    out_meta = tmp_path / "sample_metadata.parquet"

    all_df.to_parquet(in_path)

    pred_df, meta_df = run_batch(
        input_parquet=str(in_path),
        out_pred_parquet=str(out_pred),
        out_meta_parquet=str(out_meta),
        B_boot=20,
        energy_enable=False,
        n_jobs=1,
        verbose=False,
    )

    print("Batch demo:")
    print(f"  processed series: {len(pred_df)}")
    print(f"  predictor columns: {len(pred_df.columns)}")


def main() -> int:
    try:
        validate_config()
    except Exception as e:
        print(f"Config validation failed: {e}")
        return 1

    single_series_demo()

    tmp_dir = THIS_DIR / "_tmp_example"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    batch_demo(tmp_dir)

    # Clean up
    for p in tmp_dir.glob("*.parquet"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    print("Example finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
