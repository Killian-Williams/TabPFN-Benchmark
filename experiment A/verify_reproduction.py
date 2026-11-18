"""Compare two smoke run roots on memberships/probabilities, excluding timings."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_a import SPECS, digest


def compare(reference, reproduction, tolerance=1e-10, datasets=None):
    rows = []
    for dataset in (datasets or SPECS):
        left, right = Path(reference)/dataset, Path(reproduction)/dataset
        configs = [json.loads((p/"configuration.json").read_text()) for p in (left, right)]
        if not all(c["smoke"] for c in configs):
            raise ValueError("This bounded-memory verifier supports smoke bundles only")
        for config, folder in zip(configs, (left, right)):
            if config["status"] != "complete":
                raise ValueError(f"Incomplete {folder}")
            for name, expected in config["output_sha256"].items():
                if digest(folder/name) != expected:
                    raise ValueError(f"Changed evidence: {folder/name}")
        for field in ("source_sha256", "split_seeds", "sample_seeds", "percentages", "variant", "smoke", "versions"):
            if configs[0][field] != configs[1][field]:
                raise ValueError(f"Protocol/environment mismatch in {dataset}: {field}")
        keys = ["split_seed", "sample_seed", "training_percentage", "model", "source_index"]
        # This verification command targets smoke bundles; full-run comparisons should stream.
        frames = [pd.read_csv(p/"predictions.csv", dtype={"group": str}).sort_values(keys).reset_index(drop=True) for p in (left, right)]
        if any(f.duplicated(keys).any() for f in frames):
            raise ValueError("Duplicate prediction keys")
        pd.testing.assert_frame_equal(frames[0].drop(columns="probability"), frames[1].drop(columns="probability"))
        memberships = [pd.read_csv(p/"membership.csv", dtype={"group": str}) for p in (left, right)]
        pd.testing.assert_frame_equal(*memberships)
        difference = float(np.max(np.abs(frames[0].probability - frames[1].probability)))
        rows.append(dict(dataset=dataset, prediction_records=len(frames[0]),
                         maximum_absolute_probability_difference=difference, tolerance=tolerance,
                         passed=bool(np.isfinite(difference) and difference <= tolerance)))
    result = pd.DataFrame(rows)
    result.to_csv(Path(reproduction)/"reproduction_comparison.csv", index=False)
    print(result.to_string(index=False))
    if not result.passed.all():
        raise ValueError("Probability reproduction tolerance exceeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("reproduction", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=list(SPECS))
    args = parser.parse_args()
    compare(args.reference, args.reproduction, datasets=args.datasets)
