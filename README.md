`DATASET_UPDATE.md` files supersede older dataset descriptions in their README
and status records. Start with [A's dataset revision](experiment%20A/DATASET_UPDATE.md).
In particular, Olist predicts canceled versus delivered orders, not late delivery.

Hotel and shoppers are retrospective classification tasks. Shoppers has month
labels but no year or full dates; D's Sep–Dec cohort is not a verified
chronological forecast. A–C and D intentionally use different hotel feature
sets and evaluation populations. See each experiment's protocol for details.

## Repository layout

```text
data/                 Local CSV inputs; setup instructions are versioned
experiment A/         Shared baseline, data adapters, tests, and verification
experiment B/         Missingness experiments (unfinished)
experiment C/         Class-imbalance experiments (unfinished)
experiment D/         Distribution-shift experiments (unfinished)
requirements.txt      Top-level dependency constraints
```

Keep these directory names and their relative locations intact. Experiments
share helpers across folders, including imports from Experiment A. Keep the
dataset subdirectories, including legacy-named helper files, with the runners.

## Setup

Run commands from the project root. Existing verification records used Windows
and Python 3.12.10. Python 3.12 is the documented baseline.

For the recorded Experiment A environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r "experiment A\requirements-lock.txt"
.\.venv\Scripts\python.exe -m pip check
```

The root `requirements.txt` provides broader dependency constraints; it is not
an exact environment lock. B–D also have experiment-specific lock files.
Consult their instructions when reproducing their environments. The historical
validation records are not a new installation test of this upload.

The code selects the public TabPFN v2 model with CPU execution. Weights are not
bundled. First use may require a download or an existing model cache; satisfy
any access requirements reported by the installed TabPFN package. Keep access
tokens and downloaded weights out of Git.

## First execution check

Place the required data in `data/`, then run a small Experiment A check:

```powershell
.\.venv\Scripts\python.exe "experiment A\benchmark_a.py" --dataset shoppers --smoke --bootstrap 100 --output "experiment A\runs\initial_shoppers_smoke"
```

Use a fresh output directory for each independent run. Smoke mode checks the
pipeline and does not replace the full study. To inspect available options:

```powershell
.\.venv\Scripts\python.exe "experiment A\benchmark_a.py" --help
```

Follow the experiment README files for full-run commands, tests, audits,
analysis, and reproduction checks. Read `EXPERIMENT_STATUS.md` in A–C and
`REPRODUCTION_STATUS.md` in D alongside their dataset revision documents;
historical verification claims apply only to the runs and revisions they name.

## Results and initial upload policy

The initial upload prioritizes source, tests, dependency files, and documentation.
The `.gitignore` excludes local datasets, environments, caches, run directories,
bulk generated artifacts, logs, and preservation archives. These files remain
on the local machine. Local historical outputs referenced by older documents
may therefore be absent from a fresh clone.

Publish curated results separately when their completion and verification are
established. Include the generating code revision, dataset identities/hashes,
configuration, provenance, and a verification summary with each result set.
Small curated artifacts can be placed under a dedicated `results/` directory;
the ignore rules do not globally exclude CSV or image files.

Dataset and model terms are separate from any license chosen for this project's
code. No code license is selected by this preparation step.
