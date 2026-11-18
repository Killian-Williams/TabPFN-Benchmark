"""Profile the revised full design without fitting models or altering source data."""
import json
from pathlib import Path

import pandas as pd

import benchmark_a as b


def main():
    output = b.HERE / "audit_resort_shoppers_20261109"
    output.mkdir(exist_ok=False)
    datasets, splits, subsets = [], [], []
    for name, spec in b.SPECS.items():
        print("Auditing", name, flush=True)
        module, frame, profile = b.load_dataset(name)
        datasets.append(dict(dataset=name, source=spec[1], source_sha256=b.digest(b.ROOT / "data" / spec[1]),
                             eligible_rows=len(frame), positives=int(frame._y.sum()), groups=frame._group.nunique(),
                             primary_metric=spec[4], profile=json.dumps(profile)))
        for split_seed in (42, 43, 44):
            train, test = b.split_data(frame, name, split_seed)
            test_membership = b.membership(test, "test", split_seed).sort_values("source_index")
            path = output / f"{name}_test_{split_seed}.csv"
            test_membership.to_csv(path, index=False)
            splits.append(dict(dataset=name, split_seed=split_seed, train_available=len(train),
                               test_rows=len(test), test_positives=int(test._y.sum()), test_groups=test._group.nunique(),
                               group_overlap=len(set(train._group) & set(test._group)),
                               membership_file=path.name, membership_sha256=b.digest(path)))
            for seed in (101, 102, 103):
                for percentage, sample in b.nested_samples(train, seed):
                    subsets.append(dict(dataset=name, split_seed=split_seed, sample_seed=seed,
                                        percentage=percentage, rows=len(sample), positives=int(sample._y.sum())))
        model = dict(module.build_models())[b.REFERENCE].steps[-1][1]
        path = Path(model.model_path)
        if path.is_file():
            b.write_json(output / "tabpfn_checkpoint.json", dict(path=str(path), sha256=b.digest(path)))
    pd.DataFrame(datasets).to_csv(output / "dataset_manifest.csv", index=False)
    pd.DataFrame(splits).to_csv(output / "full_design_splits.csv", index=False)
    pd.DataFrame(subsets).to_csv(output / "full_design_training_counts.csv", index=False)
    print(pd.DataFrame(splits).drop(columns=["membership_file", "membership_sha256"]).to_string(index=False))


if __name__ == "__main__":
    main()
