# Allocation record index

Every JSON record in this directory is identified below. Accounting files contain segment charges; terminal files report the cumulative charge for that root, including inherited charges where present. All numbers below are transcribed from the records or summed from their segment charges. The JSON bytes and values are unchanged.

| Record | Setting | Role | Own segment charge (seconds) | Inherited charge (seconds) | Terminal cumulative charge (seconds) | Terminal GPU-hours |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| [01_accounting.json](01_accounting.json) | `R65-original` | Segment accounting. Preliminary calibration; storage projection refused training (ERROR). | 46.12347084842622 | 0 | — | — |
| [01_terminal.json](01_terminal.json) | `R65-original` | Root terminal summary. Preliminary calibration; storage projection refused training (ERROR). | — | — | 46.12347084842622 | 0.01281207523567395 |
| [02_accounting.json](02_accounting.json) | `R65-original` | Segment accounting. Continuation after root 01; calibration and two training segments; ERROR. | 3205.4585751937702 | 46.12347084842622 | — | — |
| [02_terminal.json](02_terminal.json) | `R65-original` | Root terminal summary. Continuation after root 01; calibration and two training segments; ERROR. | — | — | 3251.5820460421965 | 0.9032172350117212 |
| [03_accounting.json](03_accounting.json) | `R65-original` | Segment accounting. Preliminary calibration after roots 01–02; storage projection refused training (ERROR). | 46.31433189753443 | 3251.5820460421965 | — | — |
| [03_terminal.json](03_terminal.json) | `R65-original` | Root terminal summary. Preliminary calibration after roots 01–02; storage projection refused training (ERROR). | — | — | 3297.896377939731 | 0.9160823272054808 |
| [04_accounting.json](04_accounting.json) | `R65-original` | Segment accounting. Precursor training trajectory ending FLAT at update 102,400; inherited by the retained run. | 39323.41610861011 | 3297.896377939731 | — | — |
| [04_terminal.json](04_terminal.json) | `R65-original` | Root terminal summary. Precursor training trajectory ending FLAT at update 102,400; inherited by the retained run. | — | — | 42621.31248654984 | 11.839253468486067 |
| [05_accounting.json](05_accounting.json) | `R524` | Segment accounting. Charged preliminary calibration root; storage projection refused training (ERROR). | 50.257795182056725 | 0 | — | — |
| [05_terminal.json](05_terminal.json) | `R524` | Root terminal summary. Charged preliminary calibration root; storage projection refused training (ERROR). | — | — | 50.257795182056725 | 0.013960498661682424 |
| [06_accounting.json](06_accounting.json) | `R524-B4096-D16-short` | Segment accounting. Charged preliminary calibration root for the short deep setting; storage projection refused training (ERROR). | 157.7097463393584 | 0 | — | — |
| [06_terminal.json](06_terminal.json) | `R524-B4096-D16-short` | Root terminal summary. Charged preliminary calibration root for the short deep setting; storage projection refused training (ERROR). | — | — | 157.7097463393584 | 0.043808262872044 |
| [07_accounting.json](07_accounting.json) | `R262` | Segment accounting. Charged preliminary calibration root; storage projection refused training (ERROR). | 41.249642917886376 | 0 | — | — |
| [07_terminal.json](07_terminal.json) | `R262` | Root terminal summary. Charged preliminary calibration root; storage projection refused training (ERROR). | — | — | 41.249642917886376 | 0.011458234143857327 |

Roots 01–04 belong to the R65-original lineage. Its retained [terminal record](../R65-original/terminal.json) already includes root 04 and its ancestors. Root 02 includes root 01; root 03 includes roots 01–02; root 04 includes roots 01–03. Do not add these cumulative terminal totals together.

Roots 05–07 are the preliminary or refused roots of R524, R524-B4096-D16-short and R262, respectively. Their charges are added once to the corresponding retained-run terminal charges by [analysis/rebuild.py](../../analysis/rebuild.py). They are not extra experimental settings. Attribution follows the dated decision records and calibration chronology; the refused roots’ complete configuration manifests are not shipped. For roots 05 and 07, the calibration manifest identifiers also match calibration segments in the retained setting records; the short deep root 06 is explicitly attributed by the recorded short-schedule amendment.

A calibration may complete while training admission fails. The refused-root charges remain included in the reported total of 232.17 GPU-hours. These labels do not convert an admission failure into a training result.
