![Figure 1: Training membership and development accuracy](figures/Figure_1.png)

# Memorization and incomplete fitting in a recursive arithmetic reasoner rebuilt from its published description

This release contains the code, arithmetic datasets, saved predictions, evaluation records, and table calculations for ten exploratory training runs. The model reconstructs missing operators in arithmetic expressions. Small training files were nearly memorized; larger files remained incompletely fitted within the tested budgets. The comparisons use one training seed, 101. The motivating paper's code was not used; the implementation follows its published description.

Repository v1.0.2 is a GitHub documentation and record-label release. It clarifies device-count scopes, allocation identities and test provenance, updates citation labels, and adjusts the verifier to check the released counts. Training code, datasets, predictions and recorded measurement values are unchanged. The published Zenodo dataset remains version 1.0.1 and retains the earlier documentation and labels; no archive files were replaced.

Strict accuracy requires the complete canonical operator tuple, the chosen ordered list of operators. Semantic accuracy instead accepts any valid tuple producing the supplied result. Development rows are held-out examples reused for configuration choices and repeated evaluation; the development set contains 2,048 in-distribution (ID) rows. Out-of-distribution (OOD) rows combine a separate multiset bucket with larger target values. Multisets retain operand values and multiplicities while ignoring order.

| Measurement | Result |
| --- | --- |
| Sampled training-file fit, 65,536-row recursive / single-pass settings | 99.80% / 99.85% strict accuracy |
| Final sampled rows included / excluded from the nested training subsets | 99.60% / 33.74% at 65,536 rows; 82.75% / 49.41% at 131,072; 47.95% / 46.09% at 262,144 |
| Recorded final development, eight-step batch 4,096 / batch 1,024 settings | 1,361 / 1,002 correct rows, out of 2,048; 66.46% / 48.93% |
| Longest schedule, recorded final / largest observed development | 1,282 / 1,308 correct rows, out of 2,048 |
| Study's prespecified development threshold | 1,844 correct rows out of 2,048, the integer threshold for 90% |
| Eight retained final models, fresh-ID deficits relative to CPU development rescoring | 2.83–4.93 percentage points; 0.70–1.71 after reweighting fresh rows to the development operand-count mixture |
| Final fresh-OOD strict accuracy at each model's training recurrence count | At most 3.32% |
| Inclusive allocation charge | 232.17 GPU-hours across ten settings; at most 82.01 per setting |

A GPU-hour measures one hour of allocated GPU time. Charge includes failed and inherited allocations. The recorded development endpoints and separately executed CPU rescoring are different measurements; tables retain both. Live-training windows pool row participations on changing training weights and held state, while checkpoint fitting evaluates sampled rows using averaged weights and reset state. Updates, batch sizes, recurrence counts, and schedule clocks differ across some comparisons. These comparisons describe configuration packages.

Setting names encode training cardinality, batch, depth, and schedule. `R` denotes recursive and `S` single-pass execution. `R65-original` is the original 65,536-row file, and `R65-resampled`, `R131`, and `R262` are nested subsets of the expanded 524,288-row file. `R524` uses batch 1,024 and eight steps. Larger-batch names explicitly give `B4096`, `D8` or `D16`, and short or long schedules. Single-pass settings retain eight-update batch holding. The per-setting options and calibrated schedules are in `configs/`.

## Reproduce

The record analysis uses Python 3.10's standard library. Figure rendering additionally uses matplotlib. No model library is imported by the analysis. From the repository root, the smallest check is:

```bash
python3 -B analysis/smoke.py
```

It verifies all four included dataset hashes and executes eight canonical arithmetic rows. The full dataset and prediction payloads are provided as archive parts alongside this release. Reassemble them using the archive README. Then use the offline importer, which verifies archive hashes, every prediction hash, and all decompressed dataset hashes:

```bash
python3 -B data/fetch.py --archive-dir ../reassembled
python3 -B analysis/verify.py
python3 -B analysis/rebuild.py --output rebuilt
python3 -B analysis/check_tables.py tables rebuilt
```

`check_tables.py` requires identical file inventories and compares every CSV cell as an exact string, including float representations and empty strata. The rebuild independently executes predicted operators, checks canonical row identities, recomputes stratum counts and Wilson intervals, reconstructs all 197 live windows, and derives the displayed supplement tables. It does not read expected table cells. Table layout files contain selectors and documentary locators; measurement answer cells are computed. Method comparison, cited source results, source classes, and readout coverage are transcribed documentary records. Internal decision chronology is excluded from this public package.

The Zenodo record is published. The same verified importer uses its record file links:

```bash
python3 -B data/fetch.py --online
```

The importer uses `https://zenodo.org/records/23880859/files/<file name>?download=1`, fetches each named part, verifies its length and SHA256, reassembles it, and verifies the complete compressed and decompressed payloads. The DOI resolves to the published version 1.0.1 record. Offline reproduction is complete without online access. The same part-fetch path can be used locally with `--parts-dir ../zenodo/parts`, or pointed at a flat download directory.

To redraw all eight figures from table CSVs:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B analysis/figures.py --tables rebuilt --output redrawn
```

`src/reasoning/` contains the generator, model, optimizer, and base training implementation. `src/qdriver/` contains accumulation, single-pass, evaluation, and resumable execution code. Double-precision accumulation and checkpoint fixtures are in `src/tests/`. Training and these fixtures require a separately provisioned compatible PyTorch environment; they are shipped as source and were not executed by the record-reproduction test. The historical precision reports and negative controls are in `records/fixtures/`.

The model fixtures shipped in `src/tests/` are adaptations of the suites whose historical reports are included, written against later source versions than the earlier reports bind. Those historical reports do not attest execution of the shipped adapted bytes. The `test_provenance` section of [TRANSFORMATIONS.json](TRANSFORMATIONS.json) lists every historical source hash from the shipped suite reports that has no exact byte copy in this repository, with its historical relative filename and report locations; adaptation-source hashes remain in the per-file entries. `test_workload.py` was added for v1.0.1 and is a separate standard-library schedule check. No historical model fixture was rerun for v1.0.2.

In such an environment, source entry points include:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES="" FIXTURE_OUTPUT=fixture-output python3 -B -m unittest tests.test_c2_accumulation tests.test_checkpoint tests.test_equivalence
PYTHONPATH=src python3 -B src/train.py --config configs/R65-original.json --output outputs/example --device cpu
```

Create `fixture-output` before running the fixtures. The training command describes a full training configuration and is computationally expensive. The archive contains no model weights. Configured data hashes and complete execution records support analysis from predictions without retraining. Frozen hashing namespaces in the source preserve the exact original split and stochastic draws.

`TRANSFORMATIONS.json` records source hashes, released hashes, and exact transformation rules for adaptations and generated artifacts. Source paths use neutral aliases. Checkpoint retention columns describe the source collection at paper preparation time; no checkpoint payload is distributed. `SHA256SUMS` binds the repository files. `data/archive_manifest.json` binds the additional payloads.

[records/allocations/INDEX.md](records/allocations/INDEX.md) identifies every allocation record and its recorded charge, including the preliminary roots charged to R524, R524-B4096-D16-short and R262. The device inventory distinguishes occurrences in the full project run records (not all released) from reproducible released-record counts. `analysis/verify.py` excludes the inventory itself, scans plain and decompressed gzip records, and checks every recorded UUID against the four listed devices.

## Data and citation

GitHub repository: https://github.com/wwhite777/recursive-arithmetic-memorization

Published Zenodo dataset, version 1.0.1: https://zenodo.org/records/23880859. DOI: https://doi.org/10.5281/zenodo.23880859.

Jeong, W., & Oh, H. (2026). Memorization and incomplete fitting in a recursive arithmetic reasoner rebuilt from its published description: data, predictions and reproducible analysis (Version 1.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23880859

Repository software: Jeong, W., & Oh, H. (2026). Memorization and incomplete fitting in a recursive arithmetic reasoner rebuilt from its published description (Version 1.0.2) [Software]. https://github.com/wwhite777/recursive-arithmetic-memorization

`CITATION.cff` identifies repository software version 1.0.2 and gives the published dataset version 1.0.1 as the preferred citation and a reference, with both authors' ORCIDs. The dataset DOI identifies the archived dataset version, not repository version 1.0.2.

## Third-party notices

Upstream revisions, attribution and scope of use are in [third_party/NOTICE.md](third_party/NOTICE.md). The original [Adam-atan2 MIT](third_party/ADAM_ATAN2_LICENSE), [HRM Apache 2.0](third_party/HRM_APACHE_LICENSE) and [TRM MIT](third_party/TRM_MIT_LICENSE) notices are included there.

## Limitations

These are exploratory single-seed measurements because configuration choices followed earlier seed-101 observations. Variation across training seeds remains unresolved. Development maxima select among repeated evaluations. New evaluation populations were frozen before scoring, after development results had informed the experimental choices. The untouched-ID subset has a different operand-count mixture, so pooled comparisons require care.

The implementation differs from the published description in data generation, canonical labels, fixed recurrence counts, gradient truncation, and other documented choices. It does not reproduce the original authors' code. Batch, schedule, updates, and exposure are not all controlled in the larger-batch comparisons. The planned backward-policy comparison was not run; these results establish neither backward-policy efficacy nor diagnostics for selecting such policies.

The original small-file and expanded-file baseline final weights were deleted for storage limits before the fresh sets existed, so their fresh scores are unavailable. The longest schedule's selected-best checkpoint was not retained, with no located deletion record. Historical numerical fixtures cover accumulation and resume behavior; a numerical derivative-suffix result report was not located. Source inspection does not supply that missing result. Row-level intervals and the post hoc operand standardization do not estimate training-seed uncertainty.

Copyright (c) 2026 Woncheol Jeong. MIT license; see `LICENSE`. Authors: Woncheol Jeong (ORCID 0000-0001-9484-7808) and Hayoung Oh (ORCID 0000-0002-7362-5138).
