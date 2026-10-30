# MARVEL_GNN: failed experiments

**Written:** 2026-10-30, at project close. **Status of the project:** CLOSED.

This is a record of what did not work, so that nobody reruns it. Every number below is copied from the
report named in its section; nothing was rerun for this document. Pre-registration hashes are sha256 of the
`PREREGISTRATION.md` in the named folder.

## 1. GNN outlier head

**Hypothesis.** A GNN on the spectroscopic network, trained on synthetic corruption of CO, ranks bad lines
better than residual rules.

**Setup.**
- CO skeptic review: `reports/2026-10-15_skeptic_review/` (code at 709c4f6; no pre-registration). Ran the
  shipped checkpoint on the training script's own validation samples, plus simple rules.
- Blind C2H4 test: `reports/2026-10-29_c2h4_blind/`. Pre-registration
  `2e0c51491d7b95b9c1f4d1b8dba01758dc42676b745927db83eb2e943ad0fa18`. CO-trained, QN-masked model against
  the 12C2H4 v1 curation key (544 network negations in the ground component).

**Result.**
- CO (12C17O / 13C18O), AP: GNN 0.210 / 0.167; raw |r|/u (already the GNN's third edge feature) 0.230 /
  0.269. The rule wins on both.
- C2H4, AP on network negations: GNN 0.050 [0.041, 0.060]; |r|/u 0.054 [0.044, 0.066]; studentized 0.052;
  iterative negate-and-resolve loop **0.425 [0.381, 0.478]**; random 0.011.
  - GNN − loop = −0.38 [−0.426, −0.334]. GNN − |r|/u = [−0.007, −0.002].
  - Source negations: GNN 0.021 vs loop 0.279 [0.174, 0.391].
  - The GNN's top 50 flags on the edited graph are all bridges (h = 1) with zero residual.

**Why it failed.**
- The head ranks lines almost like one-pass |r|/u (Spearman 0.78) and adds nothing to it.
- One-pass scoring cannot localise a bad line: its residual is smeared around its cycles. The loop works
  because each removal un-masks the neighbours.
- On CO the head learned that rewired topology signals a QN bump, so it flags topology (bridges), not
  frequency errors. CO frequency-shift AP was already near random (0.041 / 0.027).

**To revisit.** Don't, as a one-pass scorer. The only open variant is iterative GNN inference (re-score
after each removal), trained on corruption that resembles real negations. Step 1′ (section 5) suggests even
that would mostly re-learn the loop.

## 2. GNN uncertainty head

**Hypothesis.** A learned per-level σ is better calibrated than MARVEL's Dijkstra + bootstrap uncertainty.

**Setup.** CO skeptic review (as above); blind C2H4 test (as above, primary endpoint 2: AUROC of line
σ/unc against 827 curator `set_unc` lines vs 36,723 negatives).

**Result.**
- CO: GNN NLL 7.19 / 6.44. The project's claimed "2–3 orders of magnitude" over legacy MARVEL shrinks to
  1.3–1.8× once the deterministic WLS covariance × Birge ratio gets one fitted scale factor (9.23 / 11.42).
- CO 13C18O: all 35 leaf levels get a GNN σ smaller than the uncertainty of their only transition. That is
  physically impossible; the training target cannot see a non-redundant line's own uncertainty.
- C2H4: GNN σ_line/unc AUROC 0.739 [0.728, 0.750], nominally above |r|/u 0.700 (pre-registered win).
  **Post hoc: 1/unc alone gives 0.755**; GNN − 1/unc = [−0.019, −0.013].
- C2H4 GNN level σ collapsed to 6.85e-4 to 8.77e-4 cm-1 for every level (p5–p95 7.0–8.4e-4). Raw σ scores
  below random (AUROC 0.337).
- SO (eval only, 2026-09-18 rerun): GNN σ correlates positively with per-level badness (r = 0.63–0.78) where
  the legacy baseline correlates negatively. The mechanism was never understood. This is not evidence of a
  working head.

**Why it failed.**
- The target measures network inconsistency, not uncertainty, and excludes leaf lines' own uncertainty.
- Domain shift: C2H4's size and uncertainty mix lie outside CO's training range, and the head falls back to
  a near-constant prior. The driving feature was never identified.

**To revisit.** Don't. The exact WLS covariance (effective resistance) is fast, deterministic and
available. Step 2 (section 7) also shows the original uncertainties are already conservative after greedy
removal.

## 3. GNN label-correction head

**Hypothesis.** The GNN recovers the correct quantum-number label of a mislabelled line.

**Setup.** CO skeptic review only. Never tested on C2H4 or real errors.

**Result.**
- Top-1: GNN 97.1% / 96.7%; rule argmin_c |E_c − E_l − ν|/u (one of the GNN's own input features) 97.1% /
  97.2%.
- The project README's "strongest head" figure (88.6% / 83.6% top-1 on the recoverable subset) predates the
  review and was never compared with this rule.

**Why it failed.** On synthetic mislabels, the best candidate is the one that closes the energy, and that
residual is an input feature. The GNN reproduces the rule.

**To revisit.** Only with real (not synthetic) misassignments, and only against the argmin rule as a
baseline.

## 4. GNN orphan-linkage head

**Hypothesis.** The GNN ranks plausible merge transitions for components dropped by MARVEL's size cutoff.

**Setup.** CO skeptic review; QN-masked retrain in the blind C2H4 report. Never tested on C2H4.

**Result.**
- MRR: GNN 0.128 / 0.052; rule "|ΔJ| = 1 first" 0.573 / 0.094 (4.5× and 1.8× better).
- QN masking (needed for the C2H4 test) drops the GNN further, to 0.081 / 0.006.

**Why it failed.** Absolute energy is unavailable across disconnected components, so the only signal is the
selection rule, which a one-line rule encodes directly.

**To revisit.** Don't. Orphan bridging is a literature search problem (see the SO 2026-09-14 case: a genuine
connectivity gap, not a modelling one).

## 5. Step 1′: line classifier vs the iterative loop

**Hypothesis.** A tabular classifier on molecule-invariant features beats the loop at ranking real network
negations under leave-one-molecule-out, and recovers ≥ 25% of the headroom.

**Setup.** `reports/2026-10-30_classifier/`. Pre-registration
`d20e6fec1113c3205c4dba7fc51c04309d3a622708d3429eb350a6bdf0a0b521`. LR and XGBoost (xgboost 3.4.1, native
API), nested leave-one-molecule-group-out, 13 groups, 18 label-free features.

**Result.**
- Pooled AP vs loop (iv′) 0.544 [0.446, 0.645]:
  - LR 0.560, **+0.016 [−0.062, +0.111]**;
  - XGB 0.499, **−0.045 [−0.185, +0.094]**;
  - label-free loop L5 0.540.
- XGB recovers 0.126 of the headroom lines (bar 0.25; random 0.042).
- **Pre-registered decision: no escalation. No GNN, no loop + GNN hybrid.**

**Why it failed.**
- The model mostly re-learns the loop: permutation importance puts `loop_pos` at 0.29 and `log_z` at 0.16.
- Source features fit the training molecules and don't transfer: permuting them on held-out molecules costs
  0.01 AP.
- Duplicate/superseded features add +0.031 [−0.019, +0.085], and their signal (`dup_discrep`) is
  residual-like.

**To revisit.** Only with evidence from outside the line list (literature, assignments, source reliability).
The residual-invisible headroom (section 9) is not reachable from these features.

## 6. A generalisable GNN

**What was planned.** The original plan was CO first, then generalise the featurisation (padded QN block +
`j_pos`, done 2026-07-06), then CO2, SO, water. No separate plan for a molecule-general GNN was ever
written.

**What it would have needed** (recorded in the blind C2H4 report's implications; none tested):
- training corruption that resembles real negations: small frequency shifts on redundant edges, at
  realistic uncertainty spreads;
- large polyatomic networks in training (CO2 became feasible once leverages scaled, section "What
  survived");
- iterative inference.
- Step 0 adds a further requirement: inputs beyond the line list, since about 43% of headroom lines are
  invisible to residuals.

**Why it stopped.**
- Steps 0, 1′ and the blind test show the target is reached by a deterministic loop.
- What the loop misses isn't visible to residual-type inputs.
- Step 1′'s classifier, given those inputs, did not beat the loop.

**To revisit.** Only if a new input source exists for the residual-invisible negations.

## 7. Step 2: alternative cleaning methods vs greedy M1

**Hypothesis.** Some method keeps more lines and levels, with accuracy no worse, than greedy "remove the
worst |r|/u line, re-solve" (M1), or greedy is provably far from the minimum-removal optimum.

**Setup.** `reports/2026-10-30_cleaning_optimality/`. Pre-registration
`3acb8d08af323da1e7a7ec3188e8f80d56a3be213b8919389014b5eced85cc43`. 18 methods at k = 2, 3, 5; 20 runs in 13
groups; 10-fold held-out NLL; MILP at k = 3. Hypatia array 1398081 (534 completed, 1 failed).

**What failed:**

- **M6-1 (greedy + MARVEL 4 replay) and M3b-T (Tukey inflate) "wins": overturned by a scoring leak.**
  - Pre-registered k = 3: both dominate M1. M6-1 ΔNLL −0.137 [−0.218, −0.065]; M3b-T −0.026 [−0.049, −0.004],
    +1.1% lines. M6-1 was the pre-registered winner.
  - Leak: held-out lines were scored with the method's final u, which inflation had set from that line's own
    full-network residual (replay: u_i ≥ |r_i|).
  - Nested check, with inflation redone per fold on training lines only (post hoc):
    - M6-1 vs M1: +0.007 [−0.001, +0.021] (held-out u = u⁰), +0.001 [−0.003, +0.007] (u⁰ × source-median
      training factor);
    - M3b-T: +0.142 [+0.064, +0.225] / +0.128 [+0.060, +0.203], worse than M1.
  - Nothing dominates M1.
  - The same leak affects every inflating method's pre-registered score: M0, M3b-x, M5, M6-x.
- **The MILP optimality test was unsound.**
  - As computed, M1 exceeds the MILP minimum by +2.1% of lines [1.0, 3.3] (17× the MILP count), so the
    pre-registered verdict was "not near-optimal".
  - But on all 11 runs checked, the MILP's own energies violate its constraints: 8–1,164 "kept" lines above
    k·u⁰, by up to 0.125 cm-1.
  - Cause: big-M ≈ 1e5 cm-1 × HiGHS integrality tolerance ≈ 1e-6 gives about 0.1 cm-1 of slack, against u⁰
    down to 1e-6 cm-1. scipy's `milp` exposes neither an integrality tolerance nor indicator constraints.
  - HiGHS also wasn't reproducible: 16O-13C-18O was optimal on Hypatia and a "Solve error" locally;
    16O-12C-18O failed everywhere.
  - Post hoc, against the best k-consistent LS set found (MILP set + M1 finish): M1 is best or tied on 7 of
    11 runs and beaten on 4 (HOCl 54 vs 52, AlO 627 vs 617, VO 419 vs 372, VO hfs 111 vs 42).
- **M3b-H (Huber inflate).** Own-set NLL +85.6; u grows only as √z, so gross outliers keep huge z.
- **M5 (Huber + Helmert per-source scales).** ΔNLL vs M1 +0.548 [+0.052, +0.999], even with the leak in its
  favour. Most scales hit the 100 cap; per-source scales driven by a few bad lines blur the good ones. On
  C2H4, 3 of the 5 key-scaled sources were capped at 100 (key values 5, 2, 2).
- **M3b-T at k = 2: numerical breakdown.** On 16O-12C-18O the Tukey weights reach about 1e-19 (u × 3.6e9,
  weight span 1.3e25). Held-out z reaches 5e5 on 2,439 lines, and the pooled NLL becomes 1.3e5. It needs a
  weight floor before any use.
- **+R re-admission.** It recovers almost nothing: M1+R 0.4 lines per run pooled, M2+R 1.7, M3a-T+R 1.1,
  M3a-H+R 27.4, M4+R 34.8, M5+R 3.8. The effect on every metric is within ±0.002. Greedy barely swamps
  (M1 0.5 lines per run).
- **M2 (studentized), M4 (weighted L1), M3a-H/T (robust remove).** They beat M1's NLL only by removing more
  lines (Δkept −0.0038 to −0.0115), so they fail the "no fewer lines" leg. M2 costs about 8× M1's runtime.
- **M6-0 (inflate only, no removal).** ΔNLL +0.563 [+0.172, +1.012].

**To revisit.**
- Greedy's optimality on VO-like runs, with a sound exact solver (tight per-line big-M or indicator
  constraints) and a multi-start greedy.
- Nothing else.

## 8. Side findings

- **`core.solver` is inexact on TiO.** `solve_energies` misses the LS minimum: χ² 1,109,487 vs 1,109,146
  with one refinement step (+0.03%), and slightly on CaOH. Step 0's TiO one-pass AP is unaffected (0.4106
  exact vs 0.4115). `core.solver` was not changed.
- **Run count: 20, not 21.** Steps 0 and 1′ and the step 2 pre-registration said 21 outlier-usable runs;
  their own list sums to 20 (12 single runs, CO2 × 6, CO × 2). All numbers are over the 20.
- **Oracle tests skip silently.** At close, `pytest` gives 44 passed, 32 skipped, 0 failed. Every skip is
  "oracle data not present": the tests point at `C:\Code\MARVEL\molecules\...`, and the parent repo is now
  `C:\Code\MARVEL_scraping`.

## What survived

- **Sparse selected-inversion leverage** (`core/selinv.py`, used by `gnn/data.py`): the exact edge
  leverage by sparse LDL^T + Takahashi recursion. 12C2H4 v1 takes 0.36 s, against 24.4 s and 5.0 GiB dense.
  SciPy only.
- **Greedy |r|/u (M1) as the cleaning baseline.** Nothing tested beats it on lines kept at equal accuracy.
  It finds 47% of C2H4's curator network negations in its first 641 removals, and it is best or tied with
  the best set found on 7 of 11 runs. At k = 3 it is the recommended triage. This now lives in
  MARVEL_network (`pyfit.py` greedy mode).
- **The headroom finding** (step 0, `reports/2026-10-30_dbmarvel_headroom/`, pre-registration
  `d84f8a6d6924b3d8d6f4126e24b2896499da05d46a95b28085db7ed83571b75c`):
  - the loop explains 0.66 of real network negations;
  - the headroom it misses is 0.27 [0.19, 0.34];
  - of those headroom lines, 1,142 of 2,633 (43%) have one-pass |r|/u < 3: invisible to any residual-type
    method.
- **Uncertainties are already conservative after greedy removal.** M1 with the original u⁰ covers 0.82 of
  held-out lines within 1σ (nominal 0.68) and 0.95 within 2σ (nominal 0.95). Inflation (replay, robust
  weights, source scales) adds nothing on held-out lines.
