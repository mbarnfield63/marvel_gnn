# Scalable edge leverages — dense timing, caching, exact sparse route

**Date:** 2026-10-29
**Requested by:** VaultAgent session (vaultagent-f7), on the user's behalf, as the blocker for running
the GNN on the 12C2H4 MARVEL v1 network. Steps 1-3 implemented/measured; options 4-6 assessed only.
Each step was reviewed and accepted by vaultagent-f7.
**Scope guard:** timing only on C2H4 — no GNN heads run, no retrain. Blind-test answer key
(overrides, decisions.md, 16AlBoRoBo corrigendum) not opened.

## Outcome

`edge_leverages` (`src/marvel_gnn/gnn/data.py`) now computes the exact leverage with a sparse
LDL^T factorization plus Takahashi selected inversion. No new dependency (SciPy SuperLU only).
Same signature, so it drops in. The dense inverse survives only as a reference in a test and in
`scripts/bench_leverage.py`.

| network (largest component) | levels | lines | nnz(L) | dense (numpy) | sparse |
|---|---|---|---|---|---|
| 12C2H4 v1 edited (deliverable) | 11,580 | 49,615 | 69,945 | 24.4 s, 5.0 GiB | 0.36 s, <10 MB |
| 12C2H4 unedited (inputs at 0b56238) | 11,681 | 50,203 | — | — | 0.35 s |
| CO2 628 main | 8,787 | 33,378 | 39,248 | (runs; used as reference) | 0.19 s |

`build_graph` total on the edited C2H4 graph: 0.88 s, down from 26 s.

## Step 1 — dense as is (accepted: eval-only OK, not for training)

Script: `scripts/bench_leverage.py --gpu`. Peak RAM via tracemalloc.
Unedited numbers in this table are from the first copy of the inputs, taken from MARVEL_network
HEAD 7b06391: 12,168 levels and 51,147 lines (see Provenance).

| | edited | unedited (HEAD copy) |
|---|---|---|
| numpy dense, local 22-core | 24.4 s, 5.0 GiB | 25.3 s, 5.5 GiB |
| torch float64, RTX 2070 | 18.5 s, 5.0 GiB VRAM, max \|dh\| 5e-11 | 21.4 s, 5.5 GiB, 1e-10 |
| torch float32, RTX 2070 | 0.9 s, 2.5 GiB, **max \|dh\| 1.0e-2** | 1.0 s, 2.8 GiB, **5.4e-2** |
| numpy dense, Hypatia (8 CPUs) | 23.3 s | 31.1 s |
| torch float64, Hypatia L40S | 5.5 s | 4.2 s |
| torch float32, Hypatia L40S | 0.4 s, max \|dh\| 9e-3 | 0.2 s, 1.8e-2 |

The Hypatia V100s were fully allocated, so the job ran on an L40S. Run dir:
`Hypatia:~/projects/MARVEL_GNN/runs/20261029-183445`, job 1397739, script `job_bench_leverage.sh`.
float32 is rejected because the weights 1/unc^2 span about 1e10.

Side findings:
- `refit_error_matrix` is not dense in n. It holds an n x n_samples array and does one sparse
  re-solve per sample, about 0.3 s per sample on C2H4 (about 1 min for 200 samples). No change needed.
- On this Windows venv, OpenBLAS with default threads (22) makes a small dense inverse 10x slower:
  863x863 takes 0.40 s, against 0.04 s with `OPENBLAS_NUM_THREADS=4`. This no longer matters,
  because the training path has no dense inverse left. Proposal: no code change; set the
  environment variable in the shell or job script if the dense reference is ever run in bulk.

## Step 2 — cache leverage across corruptions (measured, rejected)

The brief assumed most corrupted copies would be frequency-shift only, so their clean h could be
reused. That assumption is wrong. `corrupt()` hits 5% of lines and each hit is a 50/50 FREQ_SHIFT
or QN_BUMP, so nearly every copy rewires topology:

| CO train iso | lines | corrupted per copy | copies with no QN bump | prepare_corrupted(30), dense | of which edge_leverages |
|---|---|---|---|---|---|
| 13C16O | 3,382 | 169 | 0 / 1000 | 14.5 s | 12.7 s |
| 12C18O | 1,246 | 62 | 0 / 1000 | 8.8 s | 8.0 s |
| 13C17O | 58 | 3 | 126 / 1000 | 0.08 s | 0.01 s |

Reusing h per edge would be wrong, because leverage is global. A Woodbury low-rank update needs
the dense clean inverse, at about rank 170 for 13C16O. Decision (vaultagent-f7): skip the cache
and keep the corruption mix unchanged.
With the sparse route, prepare_corrupted for 3 isotopologues x 30 copies drops from 23.4 s to 3.6 s.

## Step 3 — exact sparse leverage

h_e = w_e (G_ii + G_jj − 2 G_ij), where G is the inverse of the reduced Laplacian (level 0 pinned,
as in `solve_energies`). Only the diagonal and the edge entries of G are needed.

1. `splu(A_red, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0, SymmetricMode=True)` is
   minimum-degree ordering on A^T+A with no pivoting. On an SPD matrix this gives
   L U = L (D L^T); an assert checks `perm_r == perm_c`.
2. `_selected_inverse`: Takahashi recursion over L's pattern, columns last to first:
   z[S,j] = −Z[S,S] l, z[j,j] = 1/d_j − l·z[S,j]. Cost is about Σ nnz(col)^2, small because
   spectroscopic networks fill in little: nnz(L) is about 1.7x the Laplacian's lower triangle.
3. Gather the diagonal and edge entries. Original level i sits at factor position `perm_c[i]`
   (SuperLU's `Pr A Pc = LU`). An assert on sum(h) = n−1 catches a pattern miss.

Two implementation gotchas: `lu.L.tocsc()` comes back with unsorted indices (call
`sort_indices()`), and `perm_c` maps original index to position, not the reverse.

### Validation (`scripts/bench_leverage.py --validate`)

- **All 226 CO components** (6 isotopologues, minsize 2): sum(h) = n−1 and h = 1 on every bridge.
  Max |sparse − dense| is at most 1e-9, except 13C18O main (345 levels), where it is 3.8e-7.
- **The 13C18O gap is dense's error too.** Weights there span 4e4 to 2e14, and both routes compute h as
  w·(G_ii + G_jj − 2G_ij), which cancels. Against a cancellation-free dense reference
  h = w·|C^-1 x_e|^2 (A = CC^T):

  | main component | sparse error | dense error |
  |---|---|---|
  | 13C18O | 2.5e-7 | 3.2e-7 |
  | 12C18O | 4.0e-10 | 5.9e-10 |
  | 13C16O | 1.9e-12 | 8.5e-12 |

  Sparse is never worse than dense. The 1e-10 target can't be met on 13C18O by either route, and
  that is irrelevant for a float32 edge feature.
- **CO2 628 main** (8,787 levels; dense still handles it): max |sparse − dense| is 1.7e-10;
  sum(h) and all 2,258 bridges check out.
- `tests/test_gnn.py::test_edge_leverages_match_dense_inverse` covers a 40-level random graph
  with unc from 1e-6 to 1e-2 against the dense inverse. pytest: 44 passed, 32 skipped (unchanged skips).

Ceiling: cost follows fill, not n^3. A much denser network would need a better ordering
(METIS/AMD, which is a new dependency). Not needed for anything in scope.

## Options 4-6 (assessed only)

- **4. Spielman–Srivastava random projection.** Needs k ≈ 24 ln m / ε² Laplacian solves:
  about 26k solves, about 30 s, for ε = 0.1, and the result is only approximate.
  It loses to the exact route at every size in scope. Rejected.
- **5. Subgraph sampling.** Leverage is global, so the edges cut at a subgraph boundary turn into
  spurious bridges (h → 1), and the features would differ between training and inference.
  Not needed while message passing on about 50k edges is cheap. Rejected.
- **6. Drop leverage on large components.** This buys nothing now, since exact h takes under a
  second. The scientific question (does h help?) would be settled by a CO ablation:
  EDGE_DIM 5 vs 3 (dropping h and the studentized residual), at least 3 seeds each.
  Compare held-out outlier AP pooled, per kind and per amplitude bin, plus label-fix top-1.
  Cost: 6 full CO retrains. **On hold pending the user's decision.**
  Keep h either way, because the plan uses it to label lines "unverifiable from network structure".

## Provenance of `data/C2H4_v1/` (gitignored)

- The deliverable trans file is byte-identical to MARVEL_network's
  `deliverables/C2H4_MARVEL4_20261028/` (sha256 5f44ed4a…).
- That directory is gitignored, so no commit pins it. Its mtime (2026-10-28 16:58) and
  `network/12C2H4/final_port.log` (whose counts match exactly) fall between commits 1599268 (16:38)
  and 0b56238 (17:02, "add the 12C2H4 network (v1)"). The deliverable is therefore consistent
  with 0b56238.
- The unedited inputs were re-copied read-only via `git archive 0b56238 molecules/C2H4/input`
  (47 `*_qam.txt`). They replace a first copy from HEAD 7b06391, whose inputs were rewritten at
  762a7d0, after v1 was built. Per-file sha256 values are in `input_sha256.txt`; details are in
  `PROVENANCE.txt`.
- Re-copying changed the unedited main component from 12,168 levels / 51,147 lines (HEAD copy)
  to 11,681 / 50,203 (0b56238 copy). Leverage timing is unchanged (0.35 s).
- Six sources have blank per-line uncertainties (10LoBoLoRo, 18AsRa, 23MrMaReAr, 24BeBeReMa,
  24PeLeSuRu, 26PeLeSuRu). They are filled with 1e-3 cm-1 for timing only; the policy for the
  real eval is the user's call.
- The deliverable report's header says the v1 input came from MARVEL_scraping
  `molecules/C2H4/convert/`.
- Blind guard: the `sources.tsv` notes and two report lines leaked only that a 16AlBoRoBo
  corrigendum and per-line overrides exist. vaultagent-f7 judged that acceptable.

## Carried-over items

- The CO2 oracle tests (`tests/test_co2_oracle.py`) and `scripts/plot_validation.py` point at
  `C:\Code\MARVEL\molecules\CO2`, which no longer exists; the data now lives in
  `C:\Code\MARVEL_scraping\molecules\CO2`. The tests skip silently. Not fixed here.
- C2H4 needs MAX_QN = 15, but the trained model uses 7. The bench script sets `gdata.MAX_QN` in its
  own process only. Any real C2H4 inference needs a model with a 15-wide QN block, which means a retrain.
