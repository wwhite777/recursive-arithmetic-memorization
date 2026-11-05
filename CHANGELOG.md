# Changelog

## 1.1.0 — 2026-11-04

- Add S65f/S524f fresh-batch single-pass configurations and complete records, compact batch/epoch commitments and verification receipts.
- Port fresh sampling, held-one training, replay-tail validation and schedule support into the public numerical source.
- Add all 120 final/best prediction chunks (61,104 directly scored checkpoint-row pairs), exact scored samples and membership receipt.
- Recompute control scores, strata, gaps, execution cost, endpoints and curves; extend applicable existing tables.
- Include a CPU compact-record regression suite; rerun all shipped CPU test modules.
- Supply a changed-file archive extension with explicit unchanged imports from dataset version 1.0.1 and parts capped at 16,000,000 bytes.
- Preserve the earlier DOI as the version 1.0.1 dataset citation; do not assign it to this new version.

## 1.0.2 — 2026-11-02

- GitHub documentation and record-label release; the published Zenodo dataset stays at version 1.0.1 with its earlier labels and unchanged files.
- Distinguish full-project GPU UUID occurrences from released plain/gzip record counts; exclude the inventory itself when verifying those counts against the four devices.
- Index every allocation record by setting, role and recorded charge, including preliminary or refused roots 05–07.
- Explain that shipped model fixtures adapt later source versions than earlier historical reports bind; list historical source hashes without shipped bytes.
- Identify repository software version 1.0.2 separately from the published dataset citation, version 1.0.1; correct publication and lossless-journal transformation wording.
- Training code, data, predictions and recorded measurement values are unchanged.

## 1.0.1 — 2026-11-02

- Include third-party notices and license files, with source attribution links.
- Read the shipped schedule `steps` key, retaining `act_steps` as an alias; test every shipped schedule.
- Include the recorded four-device NVIDIA L40S inventory and verify recorded GPU UUID membership.
- Version the citation and archive metadata and reduce archive parts to at most 16,000,000 bytes.

## 1.0.0 — 2026-11-02

- Initial release, commit `a95ab090` (`a95ab090e69f4d33170ab8313102b8fbcd4bc33d`).
