# MARVEL_GNN

Python port of MARVEL (Furtenbacher, Császár & Tennyson, *J. Mol. Spectrosc.* 245, 2007; Furtenbacher & Császár, *JQSRT* 113, 2012), extended with a graph neural network targeting four specific weaknesses of the original method: uncertainty calibration, outlier detection, orphan-node linkage, and quantum-number label correction.

MARVEL inverts measured rotational-vibrational transitions into empirical energy levels with well-defined uncertainties, via a spectroscopic network (energy levels as nodes, transitions as edges). This repo faithfully ports that core algorithm, then adds a shared-encoder multi-task GNN on top.

## Status

Core port and all four GNN heads are built and validated on real published data:

- **Core port** (`src/marvel_gnn/core/`) — parser, DFS component decomposition, sparse weighted least-squares solver, legacy bootstrap+Dijkstra uncertainty baseline. Energy levels validated exactly against published MARVEL input/output pairs for **CO** (6 isotopologues) and **CO2** (12 isotopologues, ~167k transitions), including inferring a missing unit-segment file from solved residuals as an auditable provenance step.
- **Uncertainty calibration head** — beats the legacy Dijkstra+bootstrap baseline by 2-3 orders of magnitude on held-out masked-refit error, matches/beats the published uncertainty column.
- **Outlier detection head** — strong recovery of synthetic QN-mislabeling corruption; frequency-shift detection is bounded by a proven structural floor (bridge transitions carry zero residual anywhere in the network) rather than a modeling gap.
- **Orphan-node linkage head** — advisory ranked merge candidates for components dropped by MARVEL's minimum-size cutoff; bounded by the genuine unavailability of absolute energy across disconnected components.
- **Label correction head** — the strongest of the four: 88.6%/83.6% top-1 accuracy on the recoverable subset of synthetic QN-mislabeling corruption.
- **Schema-oblivious featurization** — a padded quantum-number block plus one "which token is J" index, so the same model code runs on diatomic (CO), linear-triatomic (CO2), and (pending data) asymmetric-top schemas without redesign.

Remaining before a general release: retrain CO/CO2 checkpoints under the current featurization, scale edge-leverage computation to CO2-size components, and validate the asymmetric-top schema once a water dataset is available. Full history and design rationale: [`marvel_gnn_plan.md`](marvel_gnn_plan.md).

**Scientific-integrity constraint**: orphan-linkage and label-correction outputs are model predictions, not measurements. They are always advisory — surfaced for human literature review, never auto-inserted into the solve or applied to source data.

## Layout

```
src/marvel_gnn/
├── core/   # Python port: parsing, DFS components, WLS solver, legacy uncertainty baseline
└── gnn/    # Shared-encoder multi-task GNN: the four heads above
tests/      # pytest
```

## Provenance

If you use this code, please cite the original MARVEL papers:

- Furtenbacher, T., Császár, A. G., & Tennyson, J. (2007). *J. Mol. Spectrosc.*, 245, 115–125.
- Furtenbacher, T., & Császár, A. G. (2012). *JQSRT*, 113, 929–935.
