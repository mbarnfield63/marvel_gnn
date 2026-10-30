# MARVEL_GNN — Project Plan

> **CLOSED (2026-10-30).** This project is closed. What didn't work, and why, is in
> [`reports/FAILED_EXPERIMENTS.md`](reports/FAILED_EXPERIMENTS.md); read it before rerunning anything. No GNN
> head gave a usable gain over simple deterministic rules. Three tie or lose to a one-line rule on CO; the
> uncertainty head's CO edge (1.3–1.8× in NLL) collapsed on C2H4 (below 1/unc). The useful outcome, greedy |r|/u
> triage at k = 3, now lives in MARVEL_network (`pyfit.py` greedy mode). Reopen only if a new input source
> exists for the residual-invisible curator negations (about 43% of the headroom), or if a sound exact
> solver is wanted to settle greedy's optimality on VO-like networks.

**Status:** Port validated on CO **and CO2** (12-isotopologue solve oracle). Featurization and all four heads are schema-oblivious (padded MAX_QN token block + j_pos); CO GNN heads **retrained and re-validated** under that featurization (held-out results below). **Held-out CO split changed 2026-08-24**: `13C17O` swapped out for `13C18O` (`VAL_ISOS = [12C17O, 13C18O]`), retrained, all four heads' numbers updated accordingly. Edge leverages now scale exactly to CO2/C2H4-size components (2026-10-29, sparse selected inverse), which unblocks CO2 GNN training. Validation plots (Okabe-Ito thesis style) exist for both molecules' port fidelity and CO's GNN heads; a new `progress_summary.png` (2x2 grid, one panel per head, real paired GNN-vs-baseline values) was built for the HRMS talk and is kept current with the latest retrain. README/STATUS.md reflect real project state; SO2 data available for third-schema (asymmetric-top) testing without waiting on water. **Original C++ (`Source_Code_CPP`) now builds and runs on both Windows (MinGW) and Linux (WSL2)** — three-way head-to-head (C++ / Python port / published oracle) validated on 13C16O, all agree to float precision. HRMS talk figures (MARVEL_GNN section) **complete**: `marvel_algorithm.tex` recoloured to UCL brand palette, `marvel_breaks.tex` (4-layer failure-mode figure) and `marvel_gnn.tex` (2×1 grid, square canvas) both built and iterated to final state. **SO (diatomic + spin fine structure, NQN=5) run through the built C++ `MARVEL4` binary** — conversion from raw scrape format worked out, and a real (non-synthetic) orphan-component case found in a combined multi-source SO run, diagnosed to a genuine literature-connectivity gap rather than any physical selection rule. **SO now also wired into the Python port directly (legacy/`core/` path only, GNN deferred)**: all 6 isotopologues built per-source-iteratively against a `core.parse.parse_native`-compatible pipeline, a real MHz-vs-cm⁻¹ unit-labeling bug in the raw scrape data was found and fixed, and an F1/F2/F3→N spin-component QN-remap patch (literature-sourced convention) successfully bridged several orphan sources into their main networks — see 2026-09-14 dated section. **SO now has a real 3-way C++/Python/oracle validation** (`tests/test_so_oracle.py`, covering all three of 33S16O/34S16O/36S16O) and an eval-only out-of-distribution check of the CO-trained calibration head on SO. **The original 2026-09-17 version of this (r=0.48 vs r=-0.34 on 34S16O) was retracted and redone 2026-09-18**: `MARVEL_scraping` found and fixed a real 22HeStLy F-label bug and reran their bootstrap, which reversed the original ground truth the eval was checked against. Rerun on the corrected data: GNN sigma correlates positively with real per-level badness on all three isotopologues (r=0.63-0.78), while the legacy baseline correlates negatively on all three (r=-0.49 to -0.82) — a different, not-yet-fully-understood finding, not just a cleaner repeat of the old one. See 2026-09-18 dated section and `reports/2026-09-18_SO_corrected_rerun/README.md`; **not yet communicated back to `MARVEL_scraping`** (holding by explicit user instruction). Separately, `MARVEL_scraping` offered 32S16O (real published ground truth) as a genuine SO training candidate — full plan written, execution pending explicit user go-ahead (see 2026-09-18 "Part 3"). Full SO training wiring is still not started.
**Last updated:** 2026-09-17

## Purpose

Two-phase project:

1. **Port** the MARVEL C++ codebase (`C:\Code\MARVEL\Source_Code_CPP`, MARVEL4.1, ~2,670 lines, Furtenbacher/Császár/Tennyson 2007 & 2012) to Python, faithfully enough that energy levels match the original.
2. **Extend** the ported algorithm with a graph neural network that improves on four specific weaknesses of the original method: uncertainty calibration, outlier detection, orphan-node linkage, and quantum-number label correction.

Target audience: **publication / community use**, not just a personal prototype. Marco has authorial permission/affiliation with the original MARVEL authors, covering the derivative-work question — cite the 2007 (J. Mol. Spectrosc. 245) and 2012 (JQSRT 113) papers and note provenance in the README before anything goes public.

## What the original C++ actually does (reference, from reading `MARVEL4.1.cpp` / `EL_and_TR.h`)

- Parses a whitespace-tokenized transitions file into energy-level nodes (`EL`, keyed by quantum-number assignment string) and transition edges (`TR`, with frequency + uncertainty).
- DFS to decompose into connected components ("spectroscopic networks"); components smaller than `--minsize` (default 200) are **excluded from the solve entirely** — no energy is ever computed for their nodes. This is the orphan problem GNN work targets.
- Core solve per component: sparse weighted least-squares (weight = 1/unc²) via Eigen's `SimplicialLDLT` sparse Cholesky — effectively a weighted-Laplacian energy-level fit, not itself graph-ML.
- Uncertainty, two independent estimates combined via a min-graph:
  - Dijkstra shortest-path uncertainty (edge weight = unc²) from an arbitrary root node.
  - Bootstrap: resample uncertainties ~U(1,10)×original, re-solve up to 100×, `sqrt(2)·max(|diff from median|, stdev)`.
- Outlier flagging is a fixed `|diff|/unc` ratio threshold, bucketed into WRONG / VERY BAD / VERY BAD_100 / VERY BAD_1000 — no statistical model.
- Reporting outputs: `EnergyLevels.txt`, `CheckTransitions.txt`, `BadLines.txt`, `Components.txt`, `AVTable.txt`, `reviveTRs.txt`.

## Decisions

### Port fidelity
- **Energy levels must match the C++ output exactly** (float tolerance only). Treat the current MARVEL code as ground truth for this.
- **Uncertainties are explicitly allowed to differ** — this is the seam the GNN eventually replaces. Do not treat legacy uncertainty output as something to preserve bit-for-bit.
- The legacy bootstrap + Dijkstra uncertainty calculation **is still ported**, faithfully, as a baseline/control group — needed to demonstrate the GNN is actually better calibrated, not just different. Also gives a working end-to-end tool before any GNN work starts.

### Validation oracle
- Real published data, not self-generated: **CO isotopologues** (Grigorev, Dai, Potter, Xiang, Zhang, Tennyson — ApJS 283, 2026; already in `C:\Code\MARVEL\molecules\CO\CO_isotopologues_all_{input,output}.txt`, MRT format, 6 isotopologues, ~6,000 transitions in / ~1,800 energy levels out, diatomic v,J schema).
- **CO2 and water** input/output datasets to be added to `molecules/CO2` and `molecules/water` in the parent repo (Marco to fetch) — gives coverage of all three schema types in the parent CLAUDE.md (diatomic, linear-triatomic Herzberg v1,v2,l2,v3,J, asymmetric-top v1,v2,v3,J,Ka,Kc).
- **No dataset has known-bad/unedited data** — all three are post-review, published-clean. This is a hard constraint on how outlier detection and label correction can be trained/evaluated (see below): no real labeled error examples exist anywhere in scope.

### Repo / tooling
- Own repo, `MARVEL_GNN/`, nested inside `C:\Code\MARVEL` but **not** tracked by the parent git repo (added to parent `.gitignore`) and **not** a uv workspace member of the parent project (`uv init --no-workspace`).
- Own `uv`-managed venv, local to this repo (`.venv/`), not shared with any other project.
- Layout: `src/marvel_gnn/core/` (the port: parsing, solver, legacy uncertainty baseline), `src/marvel_gnn/gnn/` (all GNN work). One shared package, not two repos — chosen because the GNN needs tight access to the network structure the port builds, and both are actively changing early on.
- Phase 1 deps only for now: `numpy`, `scipy`, `networkx`, `pytest` (dev). **`torch` and `torch-geometric` deliberately not added yet** — add when Phase 2 (uncertainty head) actually starts. `torch` is already present machine-wide (CPU-only, no CUDA) but is not yet a declared dependency of this project.
- Fixed a machine-wide footgun found while scaffolding: `UV_PROJECT_ENVIRONMENT` was permanently set (Windows user env var) to a shared `...\uv\envs\gnn_co2` folder, silently redirecting every uv project's venv to one shared environment. Removed it so `uv sync` creates a normal local `.venv` per project.

### GNN architecture
- **One shared-encoder, multi-task model**, not four separate models. A single GNN encoder produces node/edge embeddings from the spectroscopic network (energy levels as nodes, transitions as edges, quantum numbers as features); separate lightweight heads read those embeddings for each task. Chosen because data is scarce (see below) and the four failure modes are genuinely correlated (a mislabeled node tends to look like an outlier *and* an orphan).
- Library: **PyTorch Geometric** (PyG) — sits on the already-installed `torch`, standard tooling for link prediction / heterogeneous graphs / message passing, most prior art for this kind of small-scientific-graph problem.
- Compute: CPU is fine at this data scale (thousands of nodes/edges); no GPU available or needed for the prototype.

### Training data & the labeling gap
- Start on **CO only** (simplest schema: v,J). Generalize node/edge featurization to the other two schemas (variable-length QN vectors, pad/mask or schema-tagged embedding) **only after** the uncertainty head is validated on CO — don't design for heterogeneity before there's evidence the core idea works.
- Since no real bad-labeled data exists anywhere in scope, outlier detection and label correction rely on:
  - **Synthetic corruption** injected into the clean published data (known corruption → known recovery target).
  - Weak pseudo-labels from the port's own bad-line ratio-threshold flagging (`BadLines.txt` logic), once the port is running — real but noisy signal, not used for outlier eval (see below), but potentially useful during development.

### Task definitions
- **Orphan-node linkage** = cross-component merge candidates. Focus on small disconnected components/nodes that never get an absolute energy because no measured transition connects them to the main network (the actual C++ failure mode — whole components silently dropped below `minsize`). GNN predicts which main-network node an orphan is most likely the same physical level as, or plausibly connected to, using QN similarity + energy-spacing patterns learned from the main network. *Not* about leaf-node uncertainty-shrinking within an already-solved component (deferred/out of scope for now).
- **Label correction** = quantum-number misassignment on a transition (wrong v/J, or wrong v1/v2/l2/v3/J, or wrong Ka/Kc — e.g. branch P/Q/R mix-ups, off-by-one J), constrained by selection rules. *Not* isotopologue mislabeling (different failure mode, out of scope for now).
- **Outlier detection vs. label correction boundary**: outlier detection is the general "something is wrong with this transition" classifier — any cause (bad frequency, wrong uncertainty, wrong QN, duplicate, transcription error). Label correction is a narrower head that only fires on the subset of outliers where the likely cause is specifically QN mislabeling, and additionally proposes the probable correct assignment. Every label-correction flag is also an outlier flag; not every outlier is a label-correction case.

### Evaluation
- **Uncertainty calibration**: both methods —
  - Held-out cross-validation (mask real transitions, refit, check predicted uncertainty against actual error) as the primary training/validation signal, avoids circularity.
  - Coverage against the published `e_E` column (e.g. is ~68% of predicted energies within 1σ of the published value) as an independent sanity check, not used for model selection.
- **Outlier detection**: synthetic injection only (precision/recall on recovering known injected corruptions). Deliberately **not** benchmarked against the legacy `|diff|/unc` ratio-threshold heuristic — that heuristic is exactly the crude baseline the GNN is meant to improve on, and anchoring eval to it risks inheriting its blind spots.
- **Orphan linkage**: validated via synthetic held-out edges (mask a real connecting edge/component-merge, check whether the GNN recovers it as a link-prediction task) — same pattern as the uncertainty CV split.

### Scientific-integrity constraint (applies to both orphan linkage and label correction)
MARVEL's core premise is that every energy level is empirically traceable to an actual measured transition — a GNN prediction is not a measurement. Both outputs are **advisory only**:
- **Orphan linkage**: output is a ranked candidate-merge report (e.g. "orphan node X likely = main-network node Y, confidence 0.87") for a human to verify against the literature. **Never auto-inserted into the weighted least-squares solve.**
- **Label correction**: output is a flagged suggestion (e.g. "transition ref X is inconsistent by Nσ; most likely correct relabeling is J'=12→11, confidence 0.9") for a human to decide whether to edit the source data. **Never silently auto-applied**, at any confidence threshold.

### Build sequence
1. Finish and validate the Python port against CO (and CO2/water once fetched) — energy levels match, legacy bootstrap+Dijkstra baseline works. This alone is a shippable, useful tool.
2. Shared encoder + **uncertainty-calibration head only** — clearest eval story (coverage vs. published e_E + held-out CV), no synthetic-corruption tooling needed to get a first signal.
3. Add **outlier detection** head (reuses encoder).
4. Add **orphan linkage** head (reuses encoder).
5. Add **label correction** head (reuses encoder, built on top of outlier detection).

Never build multiple heads simultaneously with nothing working — one validated head before starting the next.

### Explicitly out of scope for now
- Wiring MARVEL_GNN into the parent repo's existing scraping pipeline (`CLAUDE.md` Steps 1–8, specifically the Step 7 reviewer agent). Kept fully decoupled — MARVEL_GNN depends only on published, finished datasets, not on the in-progress scraping pipeline. Revisit once MARVEL_GNN is validated; no interface designed yet.
- Isotopologue-mislabeling detection (separate failure mode from QN label correction).
- Leaf-node / low-degree uncertainty strengthening within already-solved components (separate from cross-component orphan linkage).
- Any schema beyond diatomic until the uncertainty head is validated on CO.

## Phase 2 v1: uncertainty-head design (implemented)

- **Prediction target**: per-level log(sigma) in 1e-6 cm-1 units, from the *full* solved network — i.e. the level's marginal sensitivity in the actual published graph, directly interpretable as its uncertainty.
- **Training signal** (the plan's held-out CV made concrete): mask 15% of transitions uniformly at random, re-solve the component still containing the ground level, record each surviving level's energy shift as one error draw. 200 maskings per training graph; heteroscedastic Gaussian NLL. The ground level is the shift reference (error 0 by construction) and is excluded from loss and metrics.
- **Features**: nodes — v, J, log degree, log min/median incident unc, log total weight, log energy; edges (both directions) — log unc, log freq, clipped |residual|/unc from the full solve.
- **Encoder**: 3x GATv2Conv (hidden 64) with edge features + LayerNorm; head: 2-layer MLP. Shared-encoder structure in place for later heads.
- **Split**: cross-isotopologue — train on 13C16O/12C18O/13C18O (the three large networks), validate on 12C17O/13C17O (never seen in training). Held-out maskings use a different seed.
- **Controls**: legacy Dijkstra+bootstrap uncertainty and published e_E, scored as fixed sigmas on the *same* held-out masked-refit errors.
- **Caveat (documented, accepted for v1)**: predicted sigma is calibrated to "sensitivity under 15% data removal", not an absolute error scale; the coverage-vs-published-e_E comparison is the independent sanity check on that scale.
- **v1 result (2026-07-03, 1000 epochs, seed 0)**: on held-out masked-refit errors of never-seen isotopologues, GNN NLL beats the legacy baseline by 2-3 orders of magnitude and matches/beats even the published e_E (12C17O: GNN 7.8 vs legacy 2124 vs published 8.3; 13C17O: GNN 140 vs legacy 101028 vs published 2606). All three sigma sources over-cover at 1 sigma (~90% vs 68% target) — conservative scale, good per-level ranking. Both sanity coverages 100%. Val NLL still slowly falling at epoch 1000; not tuned further on purpose.

## Phase 2 v1: outlier-detection head (implemented, partially validated)

- **Setup**: edge-level head on the shared encoder — logit = MLP([h_upper, h_lower, edge features]), trained jointly with the uncertainty head (summed loss, outlier BCE upweighted 10x, pos_weight for the ~5% class imbalance). Corruption tooling in `gnn/corrupt.py`: 5% of lines corrupted, equal odds frequency-shift (+-5..500x unc, log-uniform) vs J off-by-one on the upper level (graph rewire). Corrupted graphs restricted to their largest component before the solve.
- **Joint training does not degrade the uncertainty head** (val NLL 7.80/138.37 vs 7.81/139.67 single-task).
- **v1 result (2026-07-03)**, held-out corrupted copies of never-seen isotopologues:
  - QN-bump corruptions: AP 0.31/0.20 (random 0.04/0.05), recall 73%/81% at p=0.5 — the head learns spectroscopic regularity and catches rewires. Encouraging precursor for the label-correction head.
  - Frequency-shift corruptions: AP 0.07/0.03, recall 17%/35% — near chance. Interpretation: in low-redundancy regions (degree ~2, common in the small validation networks) the solve absorbs a shifted line into the level energy, so the residual signal the features carry mostly vanishes; small shifts on non-redundant lines are close to information-theoretically undetectable. Needs redundancy/cycle-consistency features (or amplitude-stratified eval to establish the detectability floor) rather than more training.
- Feature fix along the way: the edge residual feature was clipped at 10x unc (10x and 500x corruptions looked identical); now log-scaled, which roughly doubled overall AP.

## Phase 2 v2: redundancy features + detectability floor (2026-07-03, closes the freq-shift question)

- **New edge features** (EDGE_DIM 3 -> 5): WLS leverage h of each line (h_e = w_e x_e^T L+ x_e, computed from a dense inverse of the reduced Laplacian — components are <1k levels) and the studentized leave-one-out residual r/(unc sqrt(1-h)), which undoes the solve's absorption of a shifted line. Feature-level probe: the studentized residual *alone* barely beats the raw residual, because a shift's residual distributes around its cycle — it localizes the broken cycle, not the culprit edge; disambiguation needs the message-passing context.
- **Early stopping added**: with the richer features the joint model now overfits past ~epoch 500 (val unc NLL min 47 at 500, then climbs to 152 by 1000). train.py keeps the best-val-NLL checkpoint. N_VAL_CORRUPT raised 20 -> 50 for stable amplitude bins.
- **Uncertainty head improved**: val NLL 7.39 / 86.96 (v1: 7.80 / 138.37) — now beats the published e_E on *both* validation isotopologues (8.33 / 2605.87).
- **Freq-shift verdict — the floor is structural, not a modeling gap**: 44% (12C17O) and 52% (13C17O) of validation-network lines are exact bridges (h > 0.999); a frequency shift there produces zero residual anywhere in the graph and is provably undetectable from network consistency, by any method. Amplitude-stratified recall now rises monotonically (13C17O: 17% at 5-15x unc -> 62% at 150-500x), and large-amplitude recall (46-62%) matches the non-bridge fraction — i.e. the head catches roughly what is catchable. QN-bump AP unchanged (0.29 / 0.16). The training networks are far more redundant (3-10% bridges), so detectability is much better on realistic large networks; the sparse validation networks are the worst case.
- **Decision: accept v2 and move on to the orphan-linkage head.** Remaining freq-shift headroom (small shifts on redundant edges, culprit-vs-cycle disambiguation) is bounded by the bridge fraction and not worth more iteration now. A practical corollary for reporting: leverage is worth surfacing alongside any outlier score — h ~ 1 lines should be labeled "unverifiable from network structure" rather than "clean".

## Phase 2 v1: orphan-linkage head (2026-07-06, build step 4)
- **Setup**: link-prediction head on the shared encoder. Main and orphan components are encoded **separately** (an orphan shares no energy reference with the main network), so the head is **symmetric** in the pair — input `[h_a + h_b, |h_a − h_b|, |Δv|, |ΔJ|]`; the Δv/ΔJ features inject the ΔJ = ±1 selection-rule prior. Trained on synthetic detachments of real bridge subtrees (`detach_orphans`), neg-sampled BCE (`pos_weight = n_neg`); the deliverable is `orphan_report` — a ranked advisory merge report on the real dropped components, never inserted into the solve.
- **Result — identifiability ceiling, not a training failure**: held-out true-pair MRR ≈ 0.06–0.07, hits@5 5–20% vs ~4.6% random. The discriminator that would pin an orphan's attachment is absolute energy, which is **unavailable across disconnected components by construction** (each is solved to its own ground level), so (v, J) + local structure underdetermines the exact bridge. The pair features make the advisory ranking more plausible (same-band, nearby-J) without moving the masked-edge metric. **Decision: accept — the head is built, validated, and advisory per the plan's bar; the ceiling is structural.** Revisit whether the richer linear-triatomic schema (more QNs) gives it more signal when CO2 data lands.

## Phase 2 v1: label-correction head (2026-07-06, build step 5 — final head)
- **Setup — two stages, matching the plan's outlier↔label boundary**: the outlier head selects *which* lines are suspect (`p > 0.5`); a second head (`fix_logits`) ranks candidate upper-level relabelings for each. Candidate window = every existing level with the same v and J within ±2 of the current upper (mirrors `corrupt`'s J off-by-one QN_BUMP, with distractors). Per candidate the head sees `[h_candidate, h_lower, edge features, extra]`, `extra` = `|ΔJ vs upper|, |ΔJ vs lower|, log energy–frequency mismatch |(E_cand − E_lower) − freq|/unc`. Cross-entropy against the pre-corruption upper; joint 4th loss term (weight 10). Deliverable `correction_report` is advisory only — never applied to source data.
- **Recoverable-subset ceiling**: a QN bump can rewire onto a level that does not otherwise exist; that target is in no candidate window and is unrecoverable by ranking. Training + eval are on the recoverable subset (true upper still present), the analogue of the freq-shift floor and linkage ceiling — the metric is reported against what is structurally recoverable.
- **Result — the strongest head**: top-1 accuracy **88.6% / 83.6%**, MRR **0.94 / 0.90** on the recoverable QN-bump subset (35 / 61 lines) of the never-seen isotopologues, vs ~30% random (window ≈ 3.3–3.5). The energy–frequency mismatch is the dominant feature. On the **clean published networks** the two-stage report flags only 0 / 3 lines (its noise floor; the 3 are plausible J−1 proposals). Joint training does not degrade the earlier heads (uncertainty NLL 7.3 / 75.5, outlier AP 0.23 / 0.16).
- **Why it beats the outlier/linkage heads**: it works within one connected, solved network, so absolute energies are available and the mismatch feature is a direct physical test. The other two heads are bounded by genuinely absent information (bridge residuals, cross-component energy); this task keeps its discriminating signal.
- **Decision: accept — all five build-sequence steps complete.** The full shared-encoder multi-task GNN (uncertainty, outlier, orphan linkage, label correction) is built and validated on CO.

## Core port: mixed-unit segments, QN-oblivious parsing, CO2 dataset validated (2026-07-06)

- **CO2 dataset landed and reviewed** (`C:\Code\MARVEL\molecules\CO2`): 12 isotopologues, ~167k transitions / ~45k levels, confirmed to be optimised MARVEL input/output pairs (adjusted-uncertainty column, 714 deactivated negative-wavenumber lines). Two crossed deposits (637 missing transitions, 727 missing levels — contents swapped between the pairs) were found and fixed by Marco; all 12 now validate.
- **Schema surprise**: the assignments are **7-token CDSD notation (J v₁ v₂ ℓ₂ v₃ r e/f)** — a Fermi-resonance ranking index `r` and e/f parity on top of the Herzberg labels the plan assumed. Featurization for the linear-triatomic schema must carry 7 state labels, not 5.
- **Mixed units**: the deposited transition files mix cm⁻¹ with MHz and kHz sources (sub-mm/heterodyne blocks), and the segment files carrying the unit tags were not published. The C++ mechanism (MARVEL4.1.cpp:304–430: optional segment file mapping source tag → Hz/kHz/MHz/GHz/THz, frequency and both uncertainties divided by c = 2.99792458e10) was already ported faithfully in `parse_native`; added `infer_segments`, which **reconstructs a missing segment file** by testing each tag's lines against solved level differences under every candidate unit and emits a per-tag evidence report. The inferred segment file is then fed through the faithful mechanism — an auditable provenance artifact, never a silent in-parse conversion (same integrity bar as the advisory heads). On CO2 the inference is unanimous: every non-cm⁻¹ tag matches 100% of its checkable lines under exactly one unit and 0 under cm⁻¹ (626: 38 such tags; 628/636/828/838: 2 each; 727: 1).
- **QN-count-oblivious core**: assignments were already opaque space-joined token strings end to end (matching the C++'s string keying); `parse_native` now infers `nqn` from the token count (`tokens = 2·nqn + 4`) so no schema knowledge is needed anywhere in the core. Also added: `parse_native_levels` (native levels files, everything before the last three columns is the key) and `#`-comment skipping (636 has 41 such lines; the C++ would garbage-parse them via `atof` — bug not ported).
- **Validation**: with inferred segments, 100% of active lines reproduce E′−E″ within 3σ of the adjusted uncertainty on **all 12 isotopologues** (median residuals 5e-5–4e-4 cm⁻¹); parametrized closure test per isotopologue in `tests/test_parse.py`. Segment files + evidence report regenerated by `data/make_co2_segments.py` into `data/CO2_segments/`.

## CO2 end-to-end solve oracle passes (2026-07-06)

- **Full-pipeline oracle on CO2**: parse (with inferred segments) → DFS components → WLS solve, per isotopologue, in `tests/test_co2_oracle.py`. On **all 12 isotopologues** the solved largest component's level set **exactly equals** the deposited level file (8268/5036/8787/… levels; zero missing, zero extra), confirming the published levels are exactly the main component with ground state at 0 — same structural invariant as CO.
- **Energy agreement**: unlike CO, absolute agreement varies (worst per iso: 3.8e-10 … 1.7e-3 cm⁻¹; 627/637/638/737/738/828 at ≤2e-7, 626/628/636 at 1e-5–2e-4, 838 worst at 1.7e-3) because the deposited transition uncertainties are the *adjusted* ones while the published solve used further optimization. The invariant that holds everywhere with zero exceptions: **every level within 3× its published uncertainty**. That is the oracle's assertion; no per-iso tolerance table needed.
- Runtime: full 12-iso oracle ~2.5 s (scipy sparse solve, largest system 8787 levels / 44k lines well under a second). Suite now **69 pass**.
- **CUDA note**: GPU is used only by GNN training (`gnn/train.py` auto-selects `cuda`); verified torch 2.12.1+cu126 sees the RTX 2070. The core solve stays scipy/CPU deliberately — at these sizes the sparse direct solve is sub-second and a GPU port would be complexity with no payoff.

## Schema generalization: padded QN block + j_pos (2026-07-06)

Design chosen: **pad/mask, not schema-tagged embedding** — the lazier of the two options in the plan, and sufficient because the only schema knowledge any head actually needs is *which token is J*.

- **Featurization (`gnn/data.py`)**: assignments stay opaque token strings; `qn_array` fills a fixed `MAX_QN = 8` block (zero-padded, `e`/`f` parity → 0/1, unknown non-numeric tokens fail loudly), all tokens scaled /50 (previously v/10, J/50). One extra `nqn/MAX_QN` column lets mixed-schema training distinguish a padded zero from a real one. `NODE_DIM` 7 → 14. Edge features unchanged.
- **Heads**: `link_logits` takes the full MAX_QN |ΔQN| block (padded slots differ by 0, harmless). `corrupt()` and `labelfix` take `j_pos` — the J token's position (CO `v J`: 1, the default, so all CO call sites are unchanged; CO2 CDSD `J v1 v2 l2 v3 r e/f`: 0). Candidate windows and QN bumps operate on that token, all other tokens held fixed.
- **Tests**: `tests/test_schema.py` — 7-token graph through encoder + all heads, `j_pos=0` corruption and candidate windows, real CO2 737-component featurization smoke. Suite **74 pass**.
- **Consequences flagged**:
  - Old CO checkpoints are invalid (NODE_DIM changed) and CO head metrics must be re-validated by retraining — not yet done.
  - `edge_leverages` does a dense (n×n) inverse — fine for CO components (<1k levels) but the CO2 main components are 877–8787 levels; 626/628 at ~8k need a sparse/sampled leverage computation (or per-batch subgraphs) before CO2 training. This is the real remaining blocker for training on CO2.

## Housekeeping + roadmap (2026-07-07)

- Dead-code cleanup (unused `load_segments`/`flag_bad_lines`, `MAX_QN` 8→7 spare-slot removal, `model.py` head-builder dedup, `orphan_report`'s `top_k` collapsed to 5) committed as `7251763`. Suite re-verified at 72 pass.
- README.md rewritten from a stale "no source ported yet" placeholder to reflect actual status (core port + all four heads validated on CO/CO2, schema-oblivious featurization) plus a MARVEL citation/provenance block, ahead of any public release.
- Roadmap to deployment (public community/publication release, not a hosted service) laid out: land cleanup (done) → CO2 scalable edge leverages + retrain → CO re-validate under current featurization → third schema (SO2/water) → packaging hygiene (LICENSE, public repo hosting, dependency notes) → optional author-supplied optimized-uncertainty CO data.
- New dataset flagged: SO2 MARVEL data at `C:\Code\_raw_data_store\Triatomics\SO2\MARVEL` (626/636/646, C2v bent molecule) — same asymmetric-top schema class as water. Ready now for `j_pos=3` validation; see Open items below.

## CO GNN heads retrained under the schema-oblivious featurization + validation plots (2026-07-29)

- **Retrain**: the pre-existing checkpoint (`data/marvel_gnn.pt`) predated the padded-QN-block/`j_pos` change and no longer loaded (`encoder.convs.0`/`link_head.0` shape mismatch against the new `NODE_DIM`). Reran `gnn/train.py` unchanged (CO only, same cross-isotopologue split, 1000 epochs, best-val-NLL checkpoint at epoch 300); log at `data/train_v6.log`.
- **Held-out results (12C17O / 13C17O, never seen in training) — consistent with the pre-generalization numbers, confirming the featurization change didn't regress anything**:
  - Uncertainty: NLL 8.16 / 172.68 (legacy baseline 2123.58 / 101028.40, published e_E 8.33 / 2605.87) — still 2-3 orders of magnitude better than the legacy control, on par with published. 1σ coverage 93.3% / 89.9% (68% target, conservative both isos).
  - Outlier detection: AP 0.224 / 0.144 (random 0.039 / 0.053); QN-bump recall 75.0% / 80.2%; freq-shift still bounded by the bridge-fraction floor documented in Phase 2 v2.
  - Label correction: top-1 accuracy 82.9% / 78.7%, MRR 0.905 / 0.869 (random ~30%/29%) on the recoverable QN-bump subset.
  - Orphan linkage: MRR 0.102 / 0.226, hits@5 0% / 28.3% — still weak and inconsistent, as expected from the structural identifiability ceiling (Phase 2 orphan-linkage section); correctly advisory-only.
- **Validation plots added** (`scripts/plot_validation.py`, `matplotlib` now a real dependency): `co_validation.png` / `co2_validation.png` / `project_validation.png` (classical-port residuals vs. published energies against the oracle-test tolerances) and `co_gnn_validation.png` (the four CO GNN heads above, 2x2 panel). Plot style — `thesis_params` rcParams + 8-hex Okabe-Ito palette — reused verbatim from the user's existing `Quantum Number Prediction/GNN_CO2_QN_assignment` repo rather than redefined here. No CO2 GNN plot exists yet since there is no CO2 GNN checkpoint.
- **`STATUS.md` added**: short standalone takeaways doc (what's built / how it works / validations & tests / not yet done) alongside `README.md`, for quick handoff/reference.
- **C++ direct-comparison feasibility scoped, not built**: `Source_Code_CPP/` (a local copy now sits in this repo) has source but no project file or built exe — only stale MSVC `.obj` artifacts. Needs Eigen headers (header-only) and a compiler; no `cl.exe` present, but `g++`/`gcc` are available via the Strawberry Perl toolchain and look buildable there. Queued for whenever a real MARVEL4.1-vs-GNN head-to-head is wanted.
- **Repo hygiene note**: `Akshay/` (a mentee's parallel CO/CO2 notebooks + review notes) and `Source_Code_CPP/` are both new untracked directories in the repo, and `CLAUDE.md`/`marvel_gnn_plan.md` have never been committed to git — flagged to Marco, unresolved.

## C++ head-to-head build + three-way validation (2026-08-18)

- **Built `Source_Code_CPP` from source**, both Windows (MinGW g++) and Linux (WSL2 g++) — zero changes to the algorithm files (`MARVEL4.1.cpp`, `functions.cpp`, `EL_and_TR.h`). Two gaps filled:
  - `tokenst.cpp` was missing entirely (only a stale MSVC `.obj` sat in the dir) — reimplemented `tokenize`/`Trim`/`OpenDatafile` from the single call signature actually used (`tokenize(vector<string>&, string&, delim)`).
  - Eigen 3.4.0 headers fetched (header-only, not vendored in the repo — not committed, get your own copy per the build commands already shared with Marco in-session).
- **Found + fixed a real latent bug**: `int hasSegmentfile;` (MARVEL4.1.cpp:102) is declared uninitialized — undefined behavior whenever `--segment` isn't passed on the CLI. Manifested on Windows as an apparent 21GB-RAM "hang": garbage-truthy `hasSegmentfile` sent execution down the missing-segment-tag branch, which calls `system("pause")` — a Windows-only blocking keypress wait, not a memory leak. Fix: `int hasSegmentfile = 0;`.
- **WSL2/Linux build needed one more portability fix**: `babelconfig.h`'s MSVC-only macros (`#define snprintf _snprintf` etc., lines 19-27) are unconditional and clobber glibc's real `snprintf`/`strdup`/etc. Wrapped in `#ifdef _WIN32`.
- **Validated end-to-end on 13C16O** (largest CO isotopologue, 3716 transitions, `--minsize 200`, no bootstrap): solves in <2s on Linux (the Windows hang above was the UB bug, not a genuine performance problem). Three-way diff — C++ output, a fresh Python-port solve (`solve_energies` + `marvel_solve` with `bootstrap_iterations=0`), and the published oracle (`CO_isotopologues_all_output.txt`) — all agree on all 863/863 levels to ~1e-8 cm⁻¹ (float precision). Dijkstra shortest-path uncertainties (no bootstrap) match between C++ and Python to 2.5e-6 cm⁻¹ (0.05% relative); both diverge from the published `e_E` by up to 0.036 cm⁻¹ — this reproduces, from the original C++ itself, the exact gap the GNN uncertainty head targets. Confirms the gap is architectural (crude graph-distance metric), not a Python-port discrepancy.
- **Repo path discovery (unresolved)**: the parent repo referenced throughout this plan and `CLAUDE.md` as `C:\Code\MARVEL` no longer exists under that name — renamed to `C:\Code\MARVEL_scraping` (CO oracle data now at `C:\Code\MARVEL_scraping\molecules\CO\...`). A separate, unrelated-looking `C:\Code\MARVEL5` also exists with no `molecules/` dir. Every `C:\Code\MARVEL\...` path in `CLAUDE.md`/this plan is stale; not yet bulk-corrected — flagged for Marco to confirm the intended canonical path before a sweep.
- Also produced this session, in a different repo (not tracked here): `C:\Code\Isotopologue Extrapolation\HRMS_Talk_Outline.md` — plain-language explanation of why the GNN project exists and how it bridges the uncertainty gap, for an HRMS conference talk.

## HRMS talk figures — UCL palette recolour (2026-08-18)

Grilled (via `/grilling`) the figure plan for the MARVEL_GNN section (slides 8-10) of
`C:\Code\Isotopologue Extrapolation\Abstract_2026_HRMS_Prague\HRMS_Talk_Outline.md`,
against the companion `MARVEL_GNN_Talk_outline.md` in the same repo. Three figures scoped,
all in `C:\Code\Figures\MARVEL\`:

1. **`marvel_algorithm.tex` — recoloured in place, done this session.** Extracted UCL's
   official accessible-dataviz palette from slide 2 of
   `C:\Users\Marco\Downloads\UCL-PowerPoint-Template-UCL-Sans-2.2.pptx` (dark purple
   `#361A54`, dark blue `#12239E`, dark red `#751441`, dark green `#10433F`, light blue
   `#118DFF`, light orange `#E66C37`, light pink `#E645AB`, light green `#1AAB40`; WCAG
   pairing sequence purple/green, blue/orange, green/blue, red/pink). Final state after
   several iterative tweaks: solid dark-purple panel backgrounds (all 3 panels), white
   text/headings, light-green level nodes and level lines, Source A = light orange, Source
   B = light pink.
2. **`marvel_breaks.tex` — built** (see the follow-up dated section below for final state).
3. **`marvel_gnn.tex` — built** (see the follow-up dated section below for final state).

Naming convention locked: `marvel_algorithm.tex` / `marvel_breaks.tex` / `marvel_gnn.tex`,
all in the existing `MARVEL/` category — no new category needed.

## HRMS talk figures — marvel_breaks.tex and marvel_gnn.tex built (2026-08-18)

**`marvel_breaks.tex`**: grew from the originally-scoped 3 layers to 4 through iteration.
One `.tex`, `\layer`/`\ifnum` toggle, compiled 4x to transparent PNG (`marvel_breaks_layer{1,2,3,4}.png`)
plus a combined reference PDF/PNG/SVG. Layers: (1) clean 6-node network, (2) + grey dashed
disconnected orphan cluster, (3) + a mislabeled transition (Ground → T1's terminus) drawn as
a dark-red curve that curls back on itself and does a near-180° turn right above the terminus
before landing — visually reads as absurdly roundabout for how close the two levels actually
are — with a matching dark-red flag mark, (4) + a light-blue "new source" path that traces
most of the way back into the network (G→B→C, then partway along T6's direction) before
breaking into a dashed gap, showing a connection that was never quite measured. Two real bugs
surfaced and were fixed during the layer work: caption nodes are center-anchored by default in
TikZ, so they were overflowing the fixed-canvas strut on one side even though the strut looked
wide enough — fixed with `anchor=west` on the caption style, which is more robust than just
widening the strut. Bend-curve control points from `to[out=...,in=...,looseness=...]` paths can
also swing past a nominal strut — fixed with extra margin. All 4 layers were verified
pixel-identical via the compiled PDF's `papersize` special after every geometry change, since
visual review alone isn't reliable enough to catch sub-pixel canvas drift.

**`marvel_gnn.tex`**: ended up substantially reworked from the original 3-panel-row scope
through several rounds of visual iteration:
- Panel 1 ("Network of Transitions") became a scaled-up copy of `marvel_breaks.tex`'s full
  4-layer end state — nodes and arrows only, all T-labels and captions dropped.
- Full layout refactor from 3-across to a 2×1 grid: two equal-width boxes on top (network;
  shared-encoder-with-4-heads-fanning-right), one full-width "Output" box on the bottom
  (levels left half, dashed advisory-flags box right half). Shared `\def` geometry macros
  (`\topW`, `\topH`, `\gapX`, `\gapY`, `\botW`, `\botH`) keep every box's edges provably
  aligned instead of hand-tuned per-panel coordinates.
- Canvas is exactly square (width = height) — verified via the compiled PDF's `papersize`
  after every edit, the same technique used for `marvel_breaks.tex`'s layer consistency.
- Grey connector arrows between panels removed per request.
- Real bug found and fixed: the top row appeared to overlap the bottom panel. Root cause was
  `panel_bg`'s 10pt inner-sep padding expanding every box outward on all sides, silently
  eating most of a nominal ~0.75-0.8 unit gap down to under 0.1 units of actual clearance —
  thinner than the panels' own 10pt corner radius, which is why corners visually merged. Fixed
  by raising both the horizontal and vertical inter-panel gaps to an equal 1.2 units and
  rebalancing box dimensions to hold the square canvas. **Lesson for any future multi-panel
  TikZ figure in this style: a nominal `\path`-rectangle gap between `fit`-based panel
  backgrounds is not the real visible gap — subtract `2 * inner_sep` from it to get actual
  clearance, and keep that comfortably above the corner radius.**
- Advisory-flags box content is illustrative placeholder text (e.g. "rogue experiment:
  XXAaBbCc (omitted)") added/edited live per user request during the design pass — not a real
  data point, purely for laying out the box's visual density.

Both figures' PDF/PNG/SVG were regenerated after every edit this session; self-check clean
throughout. `README.md`'s MARVEL section already lists all three (added when
`marvel_algorithm.tex` alone was built last session) — no further README changes needed.

## SO transitions through MARVEL4 + a real orphan-component case (2026-08-21)

- **Conversion from raw scrape format**: SO scrape files (`MARVEL_scraping\molecules\SO\output\`) are tab-delimited with a header row, one uncertainty column, and 5 QN tokens per side (electronic state, v, J, F, e/f — SO's ground/excited states are triplet-Sigma/Pi with spin-spin and spin-rotation fine structure, so v/J alone isn't a unique level key). Native MARVEL format wants no header, two uncertainty columns, and a plain QN block. Conversion: strip header, duplicate the uncertainty column, NQN=5. Ran 33S16O/34S16O/36S16O (single source `22HeStLy`) through the WSL-built `MARVEL4` binary — no Python-side involvement, direct C++ binary use only.
- **Diagnosed a 2-component split** in a separate, larger combined-source SO run (Akshay's `Akshay/SO/`, 3 electronic states X/B/C, ~35k transitions, ~4.5k levels; 2683/2683 split on 36S16O). Rather than guess, rebuilt the transition graph independently (Python union-find) and tested each candidate cause directly:
  - Ortho/para: doesn't apply — SO is heteronuclear, no identical nuclei.
  - F1/F2/F3 spin sub-level, e/f parity, J-parity: all three checked and ruled out — near-identical distribution of each across both components.
  - **Real cause: a genuine data-connectivity gap.** The combined file's sources apparently share no transition (or combination-difference pair) linking the two sub-networks — a real missing-bridge orphan case, not a symmetry effect or modeling artifact. `BadLines.txt`/`reviveTRs.txt` in that run dir are the next place to look for a rejected/near-miss bridging line.
- **Relevance to the plan**: this is a genuine, real-world instance of the orphan-component problem the orphan-linkage head (build-step 4) targets — a stronger validation case than the synthetic `detach_orphans` test, once/if SO is wired into the Python port and GNN pipeline (neither has happened yet; this session was C++-binary-only). Also a new real-world exercise of the pad/mask schema-oblivious featurization (NQN=5, distinct from CO2's 7-token CDSD schema) if that happens.

## HRMS talk comparison table + progress_summary figure, held-out split swap, two bugs fixed (2026-08-24)

- **HRMS deliverables**: a legacy-MARVEL-vs-GNN comparison table (one row per head) and `progress_summary.png` (`scripts/plot_validation.py`), iterated to a 2x2 grid — one panel per head, real paired bars (GNN vs its named baseline: legacy Dijkstra+bootstrap for uncertainty, random for the other three), not a normalized fold-improvement ratio, since NLL/AP/accuracy/MRR don't share a unit and a single ratio axis was judged less legible for a non-ML audience. Final styling: bold 2-line panel titles, 2-column legends centered between title and axes, no figure-level suptitle, label-correction y-axis capped at its real max of 1.0.
- **Held-out split changed on request**: `13C17O` swapped for `13C18O` (`gnn/train.py`: `TRAIN_ISOS = [13C16O, 12C18O, 13C17O]`, `VAL_ISOS = [12C17O, 13C18O]`). Retrained, best checkpoint epoch 350 (`data/train_v7_13C18O.log`). New held-out numbers (12C17O / 13C18O): uncertainty NLL 7.19/6.44 vs legacy 2123.58/51.71 (~295x/8x — the two isotopologues' legacy baselines now differ by two orders of magnitude, unlike the old split where both were large); outlier AP 0.210/0.167 vs random 0.039/0.049; label-correction top-1 97.1%/96.7% vs random 30.7%/25.9% (720 recoverable QN-bump lines on 13C18O vs 61 on the old 13C17O split — a much larger held-out sample); orphan-linkage MRR 0.128/0.052 vs an analytic random-ranking baseline `ln(n)/n` over the real candidate-pair count (109 / 884) — 0.043/0.008 — hits@1 is 0% on both isotopologues even though MRR improved, so the head stays flagged weak/advisory-only.
- **Two real bugs found and fixed while rerunning**:
  1. `gnn/train.py:233` called `.numpy()` on `graph.level_energies` without `.cpu()` first — crashes on CUDA after the checkpoint is already saved but before the outlier/label/linkage metrics print. Only surfaced now because this run selected `cuda`; earlier working retrains apparently ran CPU-only. Fixed with `.cpu().numpy()`.
  2. `gnn/train.py`'s `CO_DIR` was still `C:\Code\MARVEL\molecules\CO`, the pre-rename path (parent repo is `C:\Code\MARVEL_scraping` — flagged in this plan's open items since 2026-08-18, never actually fixed). Fixed **in `train.py` only** this session; `scripts/plot_validation.py`'s `CO_DIR`/`CO2_DIR` are still stale, so `co_validation.png`/`co2_validation.png`/`project_validation.png`/`co_gnn_validation.png` can't be regenerated without the same fix.
- **`HRMS_Notes.md` written** in the sibling repo, `C:\Code\Isotopologue Extrapolation\Abstract_2026_HRMS_Prague\HRMS_Notes.md` (outside this repo, not tracked here) — per-head speaker notes (legacy vs GNN, the number, a plain-English no-jargon explainer of how each metric is measured), a pitch script for the figure aimed at a non-ML audience, and a "what it's validated against" section (cross-isotopologue split, per-head baselines, ground truth, log/checkpoint provenance). Kept in sync with the 13C18O retrain numbers as they changed through the session.
- **Not done this session**: `co_gnn_validation.png` (the older per-isotopologue-metric 2x2 panel plot) still reflects the old 13C17O split — only `progress_summary.png`'s copy of `CO_GNN_METRICS` was updated, the two are now out of sync until `plot_validation.py`'s `CO_DIR` is fixed and that plot is regenerated.

## progress_summary.png — dashed random-chance line tried and reverted (2026-08-25)

- Added an optional `chance_line` param to `_paired_panel()` in `scripts/plot_validation.py`: a dashed black line at the top of each baseline bar, labeled "random chance", applied to panels B/C/D (not A, whose baseline is the legacy method, not chance). Regenerated and reviewed.
- Reverted on request — final `progress_summary.png` is unchanged from the 2026-08-24 version (plain paired bars, no dashed line).

## SO wired into the Python port: 6-isotopologue legacy runs, a unit bug, and a QN-remap patch (2026-09-14)

- **Scope decision**: per-isotopologue background agents built the legacy (`core/`) MARVEL path only for all 6 SO isotopologues (32S16O/32S17O/32S18O/33S16O/34S16O/36S16O) from the raw per-source scrape files in `MARVEL_scraping\molecules\SO\output\`; the GNN path was explicitly deferred (Marco's call — `gnn/train.py`'s cross-isotopologue train/val design doesn't fit a per-iso runner as-is and needs its own design pass later). Each isotopologue seeded on `22HeStLy` where available (33/34/36S16O); the other three have no `22HeStLy` source, so the seed was picked by inspecting actual QN values for the lowest-state source available. 32S16O was killed mid-run (not needed) and left unbuilt.
- **New shared utility**: `scripts/so_native_convert.py` — converts one raw scrape file (tab-separated, header row, single uncertainty column) into `core.parse.parse_native`'s native format (freq/orig_unc/optim_unc/QNs/ref), schema-oblivious across sources with different QN columns (different schemas just don't share assignment keys and don't connect — correct behavior, not a bug).
- **Real bug found**: the raw `transition_wavenumber` column is not reliably cm⁻¹ despite its name — lab/radio-astronomy rotational-spectroscopy sources (plain v,N,J schema) are actually in **MHz**, while IR/UV vibrational- or electronic-band sources are in cm⁻¹, with no unit column to distinguish them. Caught when merging 32S18O's cm⁻¹ IR source against its MHz rotational sources produced ~200,000 cm⁻¹ "residuals"; confirmed physically (82Ti's N=8 level solved at 1,571,987 "cm⁻¹" — divide by 29979.2458 MHz/cm⁻¹ and you get 52.4 cm⁻¹, exactly where SO's B≈0.7 cm⁻¹ predicts N=8 to sit). Fixed via a `freq_to_cm1` parameter on the converter, determined per-source by physical plausibility (no unit column exists to read it from). **32S17O and 32S18O's already-delivered energy files were rebuilt with the fix** — 32S16O was never built so wasn't affected; 33/34/36S16O's `22HeStLy`-seeded main networks were already correct (genuinely cm⁻¹ electronic-transition data).
- **QN-remap patch (F1/F2/F3→N)**: for SO's X³Σ⁻ ground state, the spin-component labels F1/F2/F3 map to N=J−1/J/J+1 (academic.oup.com/mnras/article/527/3/6675/7424995, Brady et al. 2024, sec. 2.2) — used to bridge orphan sources that record `(state,v,J,F,ef)` into the plain `(v,N,J)` schema used by other sources.
  - **Worked**: 32S18O's `87BuLoHa` (IR band, `ef` blank/unresolved) merged cleanly into the main network (22→172 levels after the unit fix, 4 genuine outlier lines excluded). 34S16O's `87BuLoHa` bridged its own plain-schema orphans into a new **secondary** network (181 levels/245 transitions) — not connected to the primary `22HeStLy` network, independent zero-point, written separately (`34S16O_secondary_network_*.txt`).
  - **Did not work — and why, which is itself worth recording**: bridging `22HeStLy` (the seed for 33/34/36S16O) into anything via this same patch fails, because `22HeStLy` genuinely resolves e/f parity as distinct energies (e.g. `X3Sigma- 0 1 F1 e` vs `X3Sigma- 0 1 F1 f` measured as two different levels) — unlike `87BuLoHa` where parity is unresolved/blank and safely droppable. Verified directly: 98 of 198 collapsed `(v,N,J)` keys merged two distinct original levels, and forcing it produced systematic multi-cm⁻¹ residuals across 93% of the network. Reverted cleanly, no output changed. 33S16O's other orphan (`03SaMcTh`) is separately blocked by genuine nuclear hyperfine (33S has I=3/2) — not a derivable label, real information loss to drop.
- **Deliverables**: each completed isotopologue folder (`data/SO_runs/<iso>/`) now has `<iso>_transitions.txt`, `<iso>_MARVEL_Energies.txt` (bootstrap+Dijkstra uncertainty via `uncertainty.marvel_solve`), and `report.md` with full seed/acceptance/rejection reasoning. Near-threshold residual conflicts were deliberately left in rather than hard-rejected, on the reasoning (confirmed correct) that MARVEL's bootstrap resampling is exactly the mechanism meant to absorb them into inflated level uncertainty rather than a binary accept/reject cutoff.
- **Plots**: `scripts/plot_so_energy_levels.py` — J-vs-MARVEL-energy diagrams per isotopologue, colored by electronic state, styled after Brady et al. 2024 Fig. 1 (uniform small markers, `s=10`, not size-coded). Outputs in `data/SO_runs/figures/`.
- **Not done / still open**: 32S16O never rebuilt (killed, no seed criterion resolved). 33S16O/36S16O orphans remain unpatched (genuinely blocked, not just unattempted — see above). GNN wiring for SO still not started.

## SO bridging: literature search, e/f is real doubling, C++ handoff to MARVEL_scraping (2026-09-14, cont.)

- **Literature search for a bridging source (33S16O/34S16O/36S16O)**: confirmed absent. `22HeStLy` = Heays et al. 2022 (Mol. Phys. 121:17-18, arXiv:2301.05230) — the only published B3Sigma-/C3Pi electronic-band measurement of these isotopologues; no independent second source exists. Cross-checked against VO's MARVEL precedent (Bowesman et al. 2022, MNRAS): they hit the same shape of problem (hyperfine-resolved vs. unresolved data) and shipped **three separate networks** rather than force-merging with effective-Hamiltonian synthetic transitions (considered, flagged as unreliable, not adopted).
- **Tried and abandoned**: rewrote all three isotopologues' `MARVEL_Energies.txt` to include every component in one file via a negative-energy sign convention (own script, `scripts/so_rewrite_all_components.py`) — **this was a MARVEL_GNN-side invention, not real MARVEL behavior**, and Marco chose instead to run the real `MARVEL4` C++ binary directly and format the output himself. The script/rewritten files are still on disk (harmless, gitignored under `data/`) but are not the adopted path — don't treat them as the intended deliverable in a future session.
- **`MARVEL4.1.cpp` CLI mechanics** (for whoever runs it): `--nqn` is one global value per run — cannot mix schemas in one invocation (unlike the Python port's per-line NQN inference). `--minsize` defaults to 200 — must override to 1 for these isotopologues, most real components are far smaller. `--bootstrap`/`--bootiter` has no seed control (`mt19937 gen(rd())`, reseeded every run — not reproducible, unlike the Python port's `rng=`). The binary already solves every component above `--minsize` in one run, each pinned to its own local zero, tracked via `Components.txt` — no manual multi-component handling needed once the input is schema-uniform and unit-correct.
- **Root cause of the unit bug found, upstream in `MARVEL_scraping`**: `csv_to_marvel.py:format_marvel` drops the `notes` column (and therefore the `unit: MHz` annotation extraction correctly recorded) when writing the final native `.txt`, and nothing converts the wavenumber to cm⁻¹ before that. Confirmed via `96KlSaBe_merged.csv` vs. its output `.txt`. This is why 33S16O/34S16O/36S16O's microwave sources (96KlSaBe, 03SaMcTh, 15DrHiMo, 74Ti, 82BoDeDe, 82Ti, 92LoSuOg, 03KiYa, 93Ya, 97KlBeWi) are still raw MHz in `converted/` — never actually fixed this session, unlike 32S17O/32S18O (fixed in an earlier session). **Stopgap**: hand-written `segments.txt` per isotopologue (`data/SO_runs/<iso>/segments.txt`, `<ref-tag> <unit>` pairs) matching `MARVEL4.1.cpp`'s native `--segment` format exactly — lets the C++ binary do the conversion correctly without editing the raw transitions. 32S17O/32S18O's segment files are trivial (already-fixed data, all `cm-1` except `97KlBeWi`=MHz, itself rejected/unused in 32S18O's actual solve).
- **e/f parity confirmed as real, non-derivable doubling** (not a schema-padding fix): built every `(state,v,J,F,ef)` assignment `22HeStLy` actually measured (33S16O) and checked whether `ef` is a deterministic function of `(v,N,J,F)` — it isn't; 1846/2662 derived `(v,N,J)` keys collapse two measurably-different real energies. Answers the "just add the electronic-state label to 96KlSaBe" question directly: no — state/F are trivially recoverable, but `ef` genuinely isn't, and 96KlSaBe's own schema doesn't carry it.
- **New structural finding**: 33S16O/36S16O's main-vs-sibling split (from `22HeStLy` alone) lines up almost exactly with e/f parity — e.g. F1 branch: odd J → `e` in main/`f` in sibling, even J → `f` in main/`e` in sibling, consistently. Looks like the two components are the molecule's two parity ladders, uncrossable by ordinary electric-dipole transitions (the only kind an electronic-band study can supply) — a real selection-rule wall, not a data-completeness gap. 34S16O's `22HeStLy` has the same internal 2452-vs-2452 split, previously undocumented (only its already-known 181-level secondary orphan-bridge was written up before).
- **Handoff written to `MARVEL_scraping`** (separate repo, not committed from here): `.scratch/so-multi-isotopologue-combination/PRD.md` + two issues (`01-unit-dropped-at-format-step.md`, ready-for-agent; `02-need-a-combination-agent.md`, needs-triage) + a draft new subagent `.claude/agents/combiner.md` (schema-grouping, unit-segmenting, losslessness-tested QN harmonization, connectivity analysis — never solves energies itself, never synthesizes bridging transitions). Not yet triaged/committed by the scraping-repo maintainer.

## SO/MARVEL_scraping handoff: vintage diff, real-C++ oracle test, CO-model eval on SO (2026-09-17)

- **Context**: a `MARVEL_scraping` session independently ran real `Marvel4.1.x` (C++ binary) bootstrap tests on SO's 34S16O/36S16O N/F-conflict question and wrote a cross-repo handoff (`SO_HANDOFF_from_marvel_scraping.md`, this repo's root) noting heavy overlap with this repo's own 2026-09-14 `data/SO_runs` findings (same 22HeStLy self-split, same e/f-parity-collapse root cause, independently reached via the real binary vs. this repo's Python port). The handoff proposed a 3-step order; all three were worked through this session, with user confirmation on step 3's scope.
- **Step 1 — vintage diff**: this repo's existing `data/SO_runs` reports used a `22HeStLy` source file dated 2026-07-31; `MARVEL_scraping`'s current canonical output is dated 2026-09-16 (six weeks later). Diffed both isotopologues' converted transitions by reference ID: **bit-identical** (same frequencies, uncertainties, QN assignments for all 32358/35668 refs) — the only differences were cosmetic (trailing-zero float formatting, `\r\n` vs `\n` line endings from a raw-file re-scrape). **No rerun of the existing reports needed.**
- **Step 2 — real 3-way validation, closing a plan gap**: added `tests/test_so_oracle.py`, validating this port's `parse_native`+`split_components`+`solve_energies` pipeline against `MARVEL_scraping`'s actual `Marvel4.1.x` binary output for its "fallback" (disconnected, N/F-conflict-avoiding) network builds — the same networks this repo's own 34S16O/36S16O reports independently arrived at. Agreement: ~1e-6 cm⁻¹ for 34S16O's 5028-level main network and both of 36S16O's 2683-level F1/F2 branches — same precision band as the existing CO/CO2 oracle tests. This is the first C++/Python/oracle 3-way check for SO (previously only CO had one). Oracle input/output pairs copied into `data/SO_runs/{34,36}S16O/oracle/` (gitignored, consistent with the rest of `data/SO_runs/`).
  - **Notable pitfall, not a port bug**: 10 of 34S16O's 12 sources are natively in MHz per the real run's own segment file (only `22HeStLy` and `87BuLoHa` are cm⁻¹). Parsing the fallback network without per-source unit conversion produced a ~1e7 cm⁻¹ blow-up in the WLS solve — indistinguishable at first glance from a numerical-instability bug. Ruled that out by cross-checking three independent solvers (scipy sparse LU, dense LU, `lstsq`) all agreeing with each other on the same wrong answer, which pointed at the input data rather than the algorithm; applying `parse_native`'s existing `segments` parameter (already built for exactly this case, see 2026-07-06's CO2 work) fixed it immediately.
- **Step 3 — eval-only CO-trained model on SO** (scope confirmed with the user over full retraining, since SO has no published ground-truth energy levels and full training wiring needs its own design pass): `scripts/eval_so_calibration.py` runs the existing CO-trained checkpoint's uncertainty head on SO's validated networks. SO's QN schema (`state, v, N, J, F, ef`) has three token kinds `qn_array()` cannot parse as numeric — an electronic-state name, an F1/F2/F3 spin label, and a `-` structural-disconnection dummy (see the fallback networks' `build_fallback.py` construction) — so a small eval-time adapter (`_encode_so`) numerically re-encodes them before featurization; `marvel_gnn/gnn/data.py` itself is unmodified.
  - **Finding**: on 34S16O's genuinely miscalibrated network (22HeStLy's own transitions are ~60% wrong/very-bad against the real binary's own flagging), the CO-trained GNN's predicted per-level sigma has **Spearman r=0.48** against each level's real max residual/uncertainty ratio (computed from the now-validated solve — the same quantity `Marvel4.1.x`'s `BadLines.txt` reports). The **legacy bootstrap+Dijkstra baseline gets this ranking backwards** (r=-0.34) on the identical network. On 36S16O's genuinely clean 22HeStLy branches (0% wrong/very-bad), both methods correctly show near-zero correlation — no spurious flagging on a network that's actually fine.
  - **Caveat**: the GNN's predicted sigma *magnitude* does not transfer out-of-distribution (flat ~0.0005 cm⁻¹ regardless of network, vs. the legacy method's physically-scaled ~1-2 cm⁻¹) — only the relative *ranking* within a network carries real signal for an unseen molecule/schema. Not evidence the GNN is calibrated for SO in any absolute sense.
- **Not done**: full SO training (adding SO as a real training molecule alongside CO) — deferred; would need a real design decision on what "ground truth" calibration target means for SO absent published levels (candidate: `MARVEL_scraping`'s ADR-0009 precision-ordered incremental-study residual/uncertainty ratios), and a proper (not eval-time-patched) home for the QN-token encoding in `marvel_gnn/gnn/data.py` if pursued.
- **Results sent back, 2026-09-18**: wrote `SO_HANDOFF_from_marvel_gnn.md` into `C:\Code\MARVEL_scraping`'s repo root (mirroring how their original handoff landed here) summarizing all three steps above, and messaged the `MARVEL_scraping` session pointing at it.
- **Superseded later the same day** — see the dated section below. `MARVEL_scraping` found and fixed a real 22HeStLy F-label extraction bug and reran the real bootstrap; the r=0.48/r=-0.34 finding this handoff reported was built on data now known to be substantially wrong in exactly the dimension it was measuring (per-level badness). The handoff doc left in `MARVEL_scraping`'s repo is now stale and has not yet been corrected there (holding off on cross-repo write/messaging by explicit user instruction this session — see next section for the corrected re-eval, done but not yet sent).

## SO corrected-data rerun: F-label bug retraction, oracle refresh, calibration re-eval (2026-09-18, cont.)

- **Context**: a `MARVEL_scraping` session messaged this repo mid-session reporting they'd found and fixed a real bug in `molecules/SO/papers/parse_22hestly_lines.py` — 22HeStLy's F_upper/F_lower quantum numbers had been derived from a wrong fixed Sigma→F formula instead of the source file's own `branch` column digits. They reran the real Marvel4.1.x bootstrap on all three isotopologues 22HeStLy covers (`docs/adr/0010/0011/0012` in their repo) and reported the previous handoff's headline finding does not survive: 22HeStLy is now 0.00% flagged everywhere (was ~99.6% of 34S16O's badness); 36S16O's two branches (previously disconnected, 36.3%-wrong if merged) now merge cleanly to a single 5366-level, 0%-wrong network; 33S16O got its first-ever real bootstrap test (clean, 4508 levels). Verified all three ADRs exist and match the summary before acting on it.
- **Full plan written, not yet executed at time of writing**: user asked for a 3-part plan (oracle refresh, calibration re-eval, 32S16O-as-training-data design) with the third part explicitly deferred to this plan doc pending review.

### Part 1 — oracle refresh (done)

- Replaced `data/SO_runs/{34,36}S16O/oracle/` (built from the old buggy data) and added `data/SO_runs/33S16O/oracle/` (new), each now just two files copied directly from `MARVEL_scraping/molecules/SO/marvel4_results/<iso>/`: `MARVEL_trans.txt` (their "packaged for reuse" export, native two-unc-column format) and `MARVEL_states.txt` (solved levels). Much simpler than the old fixture layout (no more `transitions_raw.txt`+`segment.txt`+unc-duplication hack) — this data is already dual-unc-column and mostly-cm⁻¹.
- **Found a second, distinct bug along the way** (not the F-label bug, not yet reported to `MARVEL_scraping` — see below): several of the "packaged for reuse" `MARVEL_trans.txt` files still have some microwave sources in raw MHz despite the file claiming "MARVEL4.1.x-ready format" — `96KlSaBe` (33/34/36S16O), `15DrHiMo`/`74Ti`/`82BoDeDe`/`82Ti`/`92LoSuOg` (34S16O only). Confirmed by checking self-consistency against each file's own `MARVEL_states.txt`: raw-MHz residuals of ~1e5–1e6 cm⁻¹ collapse to ~0 once divided by 29979.2458 MHz/cm⁻¹. Their real binary run clearly used correct units internally (the reported network stats are clean) — this is specifically an export/packaging-step bug, separate from the F-label bug they already found. **`32S16O`'s packaged files (`docs/reference/32S-16O_MARVEL_*.txt`) are unaffected** — checked all 26 of its source tags, all correctly cm⁻¹ already.
- Rewrote `tests/test_so_oracle.py` to use `infer_segments` (already built in this repo for exactly this "unit tag didn't survive export" situation) to reconstruct per-source units from the published levels, rather than trusting the file or hand-writing a segment file. All three isotopologues pass: 33S16O/34S16O to ~1e-6/1e-7 cm⁻¹ (same precision band as CO/CO2's oracle tests), 36S16O to ~3e-2 cm⁻¹ — one outlier level (`B3Sigma- 4 0 F1 e`, the node directly touched by `96KlSaBe`'s bridging transitions) sits at 0.024 cm⁻¹, explained by `infer_segments` reconstructing an approximate unit factor rather than reading `MARVEL_scraping`'s own exact segment file (not part of the export) landing on the one non-redundant bridge edge (leverage ≈1, per `edge_leverages`' own documented physics — a bridge's error is absorbed entirely into the level it touches, not spread across the network). Still 2 orders of magnitude under 22HeStLy's own 0.05 cm⁻¹ uncertainty. Full `pytest` suite still green (43 passed, 32 skipped).
- **Follow-up for next contact with `MARVEL_scraping`** (not yet sent — see "not yet communicated" below): ask them to include the real segment/unit files alongside the "ready for reuse" `trans.txt` exports, rather than treating it as a bug report — would remove the 36S16O reconstruction-precision artifact and the general fragility of re-inferring units downstream.

### Part 2 — calibration re-eval on corrected data (done)

- Updated `scripts/eval_so_calibration.py`: reads the new oracle layout (`infer_segments` + `MARVEL_trans.txt`/`MARVEL_states.txt`), evaluates the **main network only** per isotopologue (component-index logic removed — no longer needed now that 36S16O is one connected network), added 33S16O as a third case, and generalized `_encode_so` to match SO's electronic-state/F-label/dummy tokens **by value, not a hardcoded index** (36S16O's corrected schema has no dummy `-` column at all, now that it's fully merged — token count/order differs per isotopologue and is no longer safe to assume).
- **Results** (Spearman r of predicted sigma vs. real per-level badness = max |residual|/uncertainty over incident transitions, from the now-doubly-validated solve):

  | Isotopologue | Real badness | GNN sigma vs. badness | Legacy sigma vs. badness |
  |---|---|---:|---:|
  | 33S16O | 0% wrong (clean) | **r=0.626** | **r=-0.486** |
  | 34S16O | 0.169% wrong (was 59.87%) | **r=0.627** | **r=-0.720** |
  | 36S16O | 0% wrong (clean, now one network) | **r=0.783** | **r=-0.816** |

  This is a **different finding than the retracted one**, not just the same result on better data. The old comparison relied on one genuinely-bad network and one genuinely-clean control, with both methods correctly near-zero on the clean one. That control case no longer exists — all three networks are now clean-ish, and on **all three**, the GNN correlates positively with real (small-scale, continuous, not just outlier-flag) residual variation while the legacy baseline correlates **strongly negatively**, clean or not. Best unverified guess at a mechanism: real per-level residual may correlate with transition-degree (hub nodes), and legacy's bootstrap-resampling variance may be *lower*, not higher, at high-degree nodes — the opposite of what "badness" would suggest. **Not verified** — flagged honestly as an open question, not asserted as fact.
- **Not yet communicated to `MARVEL_scraping`**: by explicit user instruction this session, this corrected result (and the retraction of the r=0.48/r=-0.34 finding already sitting in their repo as `SO_HANDOFF_from_marvel_gnn.md`) has **not** been written back or messaged yet. That doc is currently stale in their repo. Whoever picks this up next should send the corrected version — see this session's local writeup at `reports/2026-09-18_SO_corrected_rerun/README.md` for the full text to adapt.

### Part 3 — 32S16O as a real training molecule (plan only, not executed — awaiting user go-ahead)

`MARVEL_scraping` offered `docs/reference/32S-16O_MARVEL_states.txt`/`_trans.txt` (Brady et al. 2024, real published ground truth, 8557 levels / 50106 transitions across 26 sources) as a training candidate, plus the three corrected minor isotopologues for inference/eval-only (no published ground truth for those). Checked and confirmed usable: all 26 of 32S16O's source tags are correctly cm⁻¹ (no MHz-export bug), self-consistent against its own states file to noise level.

**Conflict with a prior recorded decision, flagged rather than silently overridden**: this repo's own state file says "32S16O: still explicitly dropped, not a future task — don't resurrect it." That traces to 2026-09-14, when 32S16O's `core/`-path build was killed mid-run because it wasn't needed for the N/F-conflict investigation (32S16O has no N/F conflict — it's the well-studied main isotopologue). It was never a scientific rejection of 32S16O itself; using it as real training data is a different proposition. **Needs explicit user confirmation before starting**, not just inferred from this offer.

Two real design problems if this goes ahead, both bigger than a config tweak:

1. **Schema mismatch.** 32S16O's QN schema is 5 tokens (`state, v, ef, N, J` — no F needed, it has no N/F sourcing conflict), while the corrected minor isotopologues now use *inconsistent* token counts of their own (33S16O/34S16O: 6 tokens including a `-` dummy for the disconnected source; 36S16O: 5 tokens, no dummy, now that it's fully merged — see Part 2 above). `marvel_gnn/gnn/data.py`'s schema-oblivious design (padded `MAX_QN` block + `j_pos`) already generalizes over token *count* and e/f parity, but not arbitrary categorical tokens (electronic-state name, F1/F2/F3) — same gap `eval_so_calibration.py`'s `_encode_so` patches at eval-time only (see 2026-09-14/17 sections), explicitly not meant to be copied into the real pipeline as-is.
2. **Training-loop wiring.** `train.py` is CO-only right now: hardcoded `CO_DIR`, `TRAIN_ISOS`/`VAL_ISOS`, `parse_mrt_transitions`/`parse_mrt_levels` (CO's MRT format) vs. `parse_native` (SO's native format). Adding SO as a second training molecule means either parametrizing `train.py` over molecule/input-format, or a second script — a real architecture decision, not a config change. Note `train.py` currently has an *uncommitted* working-tree diff (`CO_DIR` path fix, `TRAIN_ISOS`/`VAL_ISOS` split swap, a CUDA `.cpu()` fix) sitting from a prior session — worth Marco committing that before layering more training-loop changes on top.

If pursued: 33S16O/34S16O/36S16O (corrected, no published ground truth) become inference/eval-only once a real model exists — consistent with the build-sequence rule and the scientific-integrity constraint (nothing auto-applied regardless of confidence).

**Suggested order if resumed**: (1) explicit user go-ahead to un-drop 32S16O; (2) design the QN-token encoding fix in `data.py` (affects both future SO training and un-blocks removing the eval-time `_encode_so` patch entirely); (3) design/implement the multi-molecule training-loop wiring; (4) train, validate against 32S16O's own held-out split; (5) run the trained model on 33/34/36S16O inference-only, advisory outputs only.

## Exact sparse edge leverages: C2H4 / CO2 scaling (2026-10-29)

Task from VaultAgent (vaultagent-f7), blocking a planned GNN blind test on the 12C2H4 MARVEL v1 network (11,580-level ground component). Full write-up: `reports/2026-10-29_sparse_leverage/README.md`.

- **Decision: `edge_leverages` is now exact and sparse.** It factors the reduced Laplacian with SciPy SuperLU as LDL^T (`MMD_AT_PLUS_A` ordering, `diag_pivot_thresh=0`, SymmetricMode, asserts `perm_r == perm_c`). Takahashi selected inversion over the factor's pattern then yields only the diagonal and edge entries of the inverse. Same signature; no new dependency.
- **Cost:** 12C2H4 0.36 s, against 24 s and 5 GiB dense; CO2 628 0.19 s. Cost follows fill: nnz(L) is about 1.7x the Laplacian's lower triangle on spectroscopic networks.
- **Accuracy:** matches dense on all 226 CO components and on CO2 628 (≤1e-9), except 13C18O main (3.8e-7). That 1e10 weight spread makes w(Gii+Gjj−2Gij) cancel in both routes, and a cancellation-free Cholesky reference shows sparse is never worse than dense.
- **Dense is kept only as a reference** (test, plus `scripts/bench_leverage.py --validate`).
- **Rejected:** float32 GPU inverse (max |dh| about 1e-2); caching clean h across corruptions (every CO corrupted copy rewires topology via QN bumps, 0/1000 frequency-only); random-projection effective resistance (about 30 s and approximate, vs 0.36 s exact); subgraph sampling (boundary edges become spurious bridges).
- **On hold for the user:** an ablation of whether h earns its place (EDGE_DIM 5 vs 3, 6 CO retrains). Keep h for the "unverifiable from network structure" label regardless.
- **Side findings:** `refit_error_matrix` is not dense in n. OpenBLAS's default threads make small dense inverses 10x slower on this Windows venv; this is now irrelevant to training.
- **C2H4 notes:** it needs MAX_QN = 15, but the trained model uses 7, so real inference needs a retrain. Six unedited sources lack per-line uncertainties; the eval policy is the user's call. The v1 deliverable is consistent with MARVEL_network 0b56238 (`data/C2H4_v1/PROVENANCE.txt`). Blind-test answer key (overrides, decisions.md, 16AlBoRoBo corrigendum) not opened.

## QN-masked model + blind C2H4 test of the outlier and uncertainty heads (2026-10-29)

Full write-up and pre-registration: `reports/2026-10-29_c2h4_blind/`.

**Decision: QN masking.** `gnn/data.py` now has `QN_MASK = True` (QN block zeroed, token-count column
constant) and `MAX_QN = 15`. The heads see network structure only, because a CO-trained model has never seen
another schema's labels.
- CO held-out results under masking: NLL 7.14 / 6.79 and outlier AP 0.206 / 0.205, against 7.19 / 6.44 and
  0.210 / 0.167 before. The uncertainty and outlier heads never relied on labels.
- Orphan-linkage MRR drops (0.081 / 0.006).
- The old checkpoint is kept as `data/marvel_gnn_maxqn7_20260824.pt`.

**Blind test on 12C2H4 v1** (unedited 0b56238 inputs vs curation's overrides; pre-registered; hash-frozen):
- **Outlier.** The GNN loses to the simulated iterative negate-and-re-solve loop: AP 0.050 vs 0.425 on 544
  network negations, CI on the difference [−0.43, −0.33]. It also loses narrowly to one-pass |resid|/unc
  (0.054), and its ranking is close to that baseline's (ρ 0.78).
  - Rule-based negations (QAB, QAI/QAM) are invisible to every score.
  - QAC and QAJ are found only by the loop.
- **Uncertainty.** The pre-registered AUROC for line-level set_unc is 0.739, which beats the residual
  baselines. Post hoc: the GNN level σ is near-constant on C2H4 (6.9–8.8e-4 cm-1), and 1/unc alone scores
  0.755. The head carries no signal on this molecule.
- **Advisory top-50 flags:** all bridges with zero residual. The outlier head flags topology, not errors.
- **Verdict:** the CO-trained GNN does not replace the MARVEL curation loop on large polyatomic networks.
  - Candidate fixes (untested): realistic corruption mixes; training on large networks (CO2); iterative
    GNN inference; finding out why the uncertainty head collapses.
  - A cheap residual-driven iterative loop already recovers 47% of curation's network negations.

## db_MARVEL headroom study, step 0 (2026-10-30)

Task from vaultagent-26, approved by the user. No GNN involved. Write-up:
`reports/2026-10-30_dbmarvel_headroom/README.md`. Pre-registration sha256 d84f8a6d…b75c.

**Question.** How much of real curation does the no-model pipeline (iterative loop plus generic QN rules)
already recover, across db_MARVEL's 21 publications plus C2H4?

**Scope.** 13 molecule groups and 21 runs for outliers; 7 groups and 20 runs for uncertainty.

**Label audit findings:**
- db_MARVEL has no removal reasons.
- "Uncertainty raised" is mostly MARVEL 4's automatic optimisation (used ≈ the final |residual|), not a
  curator decision.
- Loader bugs in db_MARVEL: CaOH field widths, D2H+ units, N2O has no manifest. VaultAgent logs these.

**Partition of real network negations** (molecule-weighted):

| | bridge | loop | rule only | headroom |
|---|---|---|---|---|
| K from the labels | 0.07 | 0.66 | 0.01 | 0.27 [0.19, 0.34] |
| label-free loop | | | | 0.28 [0.17, 0.39] |

- Both land in the pre-registered judgement band.
- About 43% of headroom lines have |r|/unc < 3, so residuals can't see them. The rest are flagged lines that
  curators negated where the loop's budget went to lines curators inflated instead.

**Uncertainty.**
- Loop then MARVEL 4-style replay: AUROC 0.89 pooled, ≥ 0.91 in 5 of 7 groups.
- The shortfall is PN and C2H4.
- On C2H4's curator label, replay (0.87) beats the blind test's GNN (0.74).

**Methods notes (deviations from the pre-registration):**
- Leverage needs unc floored at 1e-6 cm-1. PN's 1e17 weight range breaks the factor.
- AP is tie-aware: the line-order tie-break gave a binary score with no hits AP ≈ 1.

**Recommendation (mine, for the user).** Don't run step 1 (real-label LOMO GNN) as specified. First test
negate-vs-inflate with a small tabular classifier on high-residual lines, leave-one-molecule-out, against the
loop. Only escalate to a GNN or a loop+GNN hybrid if that beats the loop.

## Step 1′: line classifier vs the iterative loop (2026-10-30)

Task from vaultagent-26, approved by the user. Write-up: `reports/2026-10-30_classifier/README.md`.
Pre-registration sha256 d20e6fec…b521.

**Setup.**
- Models: logistic regression and XGBoost (CPU, xgboost 3.4.1, a new dependency approved by the user),
  trained leave-one-molecule-group-out with nested tuning.
- Data: the headroom study's 21 runs in 13 groups.
- Features: 18 molecule-invariant, label-free features (residual, structure, source,
  duplicate/superseded).
- The loop-position feature comes from a new label-free loop, L5: stop at max |r|/unc ≤ 3, or after removing
  5% of lines.

**Result: negative.**

| pooled AP | vs loop (iv′), 0.544 |
|---|---|
| XGB 0.499 | −0.045 [−0.19, +0.09] |
| LR 0.560 | +0.016 [−0.06, +0.11] |

- The label-free L5 scores 0.540.
- XGB recovers 0.13 of the headroom H lines (random 0.04); the bar was 0.25.
- **Decision rule: no escalation. No GNN and no loop+GNN hybrid; the loop stays the tool.**

**Supporting findings.**
- Duplicate/superseded features: +0.031 [−0.02, +0.09] AP. Their signal is `dup_discrep`, which is
  residual-like.
- The model mostly re-learns the loop (permutation: `loop_pos` 0.29, `log_z` 0.16). Source features fit the
  training molecules but don't transfer.
- Untested follow-up: the one-pass residual puts half of the visible H lines in its top K. A loop that keeps
  one-pass flags (for example, ranking by the max of the one-pass and loop positions) is a cheap
  deterministic follow-up.

**Method note.** XGBoost runs through the native `xgb.train` API, because the sklearn wrapper needs
scikit-learn, which isn't approved.

## Step 2: how optimal are the cleaning methods? (2026-10-30)

Task from vaultagent-26, approved by the user. Write-up: `reports/2026-10-30_cleaning_optimality/README.md`.
Pre-registration sha256 3acb8d08…cc43 (revised with VaultAgent's +R variants and removal attribution before
any scoring run). Scripts: `scripts/cleaning_optimality.py`, `scripts/cleaning_report.py`,
`scripts/cleaning_posthoc.py`. Ran as Hypatia array 1398081 (534 CPU tasks).

**Setup.**
- 20 runs in 13 groups (not 21, as steps 0 and 1′ said).
- 18 methods at k = 2, 3, 5: curators M0; greedy M1/M2; Huber/Tukey IRLS remove or inflate; weighted L1;
  Huber + Helmert per-source scales; MARVEL 4 replay (alone or after greedy); +R re-admission variants.
- Scored by 10-fold held-out NLL on the method's own network, plus lines kept and levels validated. Pareto
  front and pre-registered dominance rule. A big-M MILP (scipy/HiGHS) at k = 3 measured greedy's gap.

**Results.**
- **Pareto front (k = 3):** M0, M6-0, M3b-T, M6-1, M5+R. M6-1 (greedy + replay) and M3b-T (Tukey inflate)
  dominate greedy M1. Pre-registered winner: M6-1 at k = 3, ΔNLL vs M1 −0.137 [−0.22, −0.07] with the same
  lines.
- **Overturned post hoc: a scoring leak.** Held-out lines carried u inflated from their own full-network
  residual. With inflation redone per fold (nested check from VaultAgent's review), M6-1 vs M1 is +0.001
  [−0.003, +0.007] and M3b-T is +0.128 (worse). Nothing dominates M1, so **greedy M1 at k = 3 is
  adequate**. The leak affects every inflating method's pre-registered score (M0, M3b, M5, M6).
- **Greedy's optimality.** The pre-registered MILP test says "not near-optimal" (+2.1% of lines), but the
  MILP is unsound at this precision. HiGHS's integrality tolerance times big-M leaves about 0.1 cm⁻¹ of slack,
  so the "optima" violate their own constraints on every run checked.
  - Post hoc, against the best k-consistent LS set found (MILP set + M1 finish), greedy is best or tied on 7
    of 11 runs and beaten on 4. The large losses are VO (419 vs 372) and VO hfs (111 vs 42). Pooled gap
    0.11% of lines.
- **Re-admission (+R) and studentized/L1/Huber variants buy nothing** over greedy + replay.
- **M3b-T needs a weight floor:** at k = 2 the Tukey weight span reaches 1.3e25 and the LS breaks.
- **`core.solver` misses the LS minimum on TiO** (χ² +0.03%). Step 0's TiO AP is unaffected (0.4106 vs
  0.4115).

**C2H4 v2 recommendation (advisory).** Use greedy M1 at k = 3 as a flag list, not an edit: 3,180 removals
(6.3%) vs the curators' 679. Leak-free common-set NLL is −6.56 vs −6.44. Only 10% of M1's removals are
curator negations; the curators inflate where M1 removes. The replay is optional bookkeeping and adds no
held-out accuracy.

## Close-out (2026-10-30)

The user closed the project. `reports/FAILED_EXPERIMENTS.md` records every negative result: the four GNN
heads, the step 1′ classifiers, the generalisable-GNN requirements, the step 2 alternatives (including the
overturned M6-1/M3b-T wins and the unsound MILP), and the side findings.

What survived:
- sparse selected-inversion leverage (`core/selinv.py`);
- greedy |r|/u as the cleaning baseline (moved to MARVEL_network, `pyfit.py` greedy mode, built by
  marvel-network-25);
- the headroom finding (43% of headroom lines are residual-invisible);
- the conclusion that uncertainties are already conservative after greedy removal.

At close, `pytest` gives 44 passed, 32 skipped, 0 failed. Every skip is oracle data missing at the stale
`C:\Code\MARVEL\molecules` path.

**Reopen only if:**
- a new input source exists for the residual-invisible curator negations; or
- a sound exact solver is wanted to settle greedy's optimality on VO-like networks.

The open items below are frozen as of close.

## Open items / next actions
All five build-sequence steps are complete on CO: the `core/` port (parser, DFS components, WLS solver, legacy bootstrap+Dijkstra baseline, energies validated against published output) and all four shared-encoder GNN heads (uncertainty, outlier, orphan linkage, label correction). The prototype is functionally complete on diatomic (v, J) data. What remains:

**Ready now (CO2 data validated):**
- [x] End-to-end CO2 solve oracle — done 2026-07-06, `tests/test_co2_oracle.py`, all 12 isotopologues pass (exact level-set match, all levels within 3σ).
- [x] **Schema generalization (featurization + heads)** — done 2026-07-06: padded MAX_QN token block + `j_pos`, all heads schema-oblivious, 7-token CDSD validated synthetically and on real CO2 (see dated section). The asymmetric-top schema (v₁v₂v₃, J, Ka, Kc) should slot in with only a `j_pos=3` — untested until the water data lands.
- [x] **Re-train + re-validate CO head metrics** under the new featurization — done 2026-07-29, see dated section above; held-out numbers consistent with pre-generalization results.
- [x] **Scalable edge leverages for CO2-size components** — done 2026-10-29: exact sparse LDL^T + Takahashi selected inverse (SciPy only). CO2 628 takes 0.19 s, 12C2H4 0.36 s. See dated section.
- [ ] **Fix the CO2 oracle test path** — `tests/test_co2_oracle.py` still points at `C:\Code\MARVEL\molecules\CO2`, so all its tests skip silently. The data is in `C:\Code\MARVEL_scraping\molecules\CO2`.
- [x] **Blind C2H4 test** of the outlier and uncertainty heads — done 2026-10-29: negative result, see dated section.
- [ ] **Understand the C2H4 uncertainty-head collapse** (near-constant σ) before any further large-network use.
- [x] **db_MARVEL headroom study (step 0)** — done 2026-10-30. Headroom is 0.27–0.28 of real network
  negations: a judgement call. See dated section.
- [x] **Step 1′ classifier test** — done 2026-10-30: negative (no model beats the loop under
  leave-one-molecule-out). The pre-registered rule says no GNN or hybrid. See dated section.
- [x] **Step 2 cleaning-optimality study** — done 2026-10-30. The pre-registered winner (M6-1, greedy + replay)
  was overturned by a held-out-u leak; leak-free, greedy M1 at k = 3 is adequate. The MILP optimality test
  turned out unsound (big-M tolerance). See dated section.
- [ ] **Optional:** settle greedy's optimality with a sound exact solver (indicator constraints or tight M),
  and try a multi-start greedy on VO-like runs, where greedy was beaten by up to 62%.
- [ ] **Optional:** M3b-T with a weight floor. Leak-free it is worse than M1 (+0.13 nats); low priority.
- [ ] **Optional, cheap:** a deterministic loop variant that keeps one-pass flags. The one-pass residual
  holds half of the visible headroom. User's call.
- [ ] **Leverage-feature ablation** (option 6, 2026-10-29): EDGE_DIM 5 vs 3, at least 3 seeds, 6 CO retrains. On hold pending the user's decision.
- [ ] **CO2 training run** — extend `train.py`'s dataset to CO2 components with `j_pos=0` (leverages now scale, 2026-10-29). The orphan-linkage head's best chance for real signal (more QNs ⇒ less underdetermined bridge problem).
- [x] **Build `Source_Code_CPP` and run MARVEL4.1 directly** — done 2026-08-18 on Windows (MinGW) and Linux (WSL2); three-way validated against Python port + published oracle on 13C16O, see dated section above. A true GNN-output head-to-head (not just port-fidelity) is still open — the C++ only reproduces the legacy Dijkstra/bootstrap baseline, already known to underperform the GNN heads.
- [ ] **Fix stale `C:\Code\MARVEL\...` paths** in `CLAUDE.md` and this plan's prose — parent repo renamed to `C:\Code\MARVEL_scraping`; confirm canonical path with Marco before a bulk sweep. **Partially done 2026-08-24**: `gnn/train.py`'s `CO_DIR` fixed (needed to rerun training); `scripts/plot_validation.py`'s `CO_DIR`/`CO2_DIR` still stale.
- [x] **Build `marvel_breaks.tex` and `marvel_gnn.tex`** in `C:\Code\Figures\MARVEL\` — done 2026-08-18, see dated section above. All three HRMS talk figures for the MARVEL_GNN section now complete.
- [x] **HRMS comparison table + `progress_summary.png`** — done 2026-08-24, see dated section above. Speaker notes in the sibling repo's `HRMS_Notes.md`.
- [ ] **Regenerate `co_gnn_validation.png`** under the new 13C18O held-out split (currently still reflects the old 13C17O split) — blocked on the `plot_validation.py` `CO_DIR` fix above.
- [ ] **Rerun `gnn/train.py` on CPU or fix the wider CUDA path** — the `.cpu().numpy()` bug (2026-08-24) suggests other CUDA-path code may be similarly unexercised; worth a once-over before the next CUDA retrain.

**Ready now (asymmetric-top data available):**
- [ ] **SO2 dataset** at `C:\Code\_raw_data_store\Triatomics\SO2\MARVEL` — 626, 636, 646 isotopologues, C2v (bent) molecule, asymmetric-top schema (v1 v2 v3 J Ka Kc), same class as water. Use this to validate `j_pos=3` and the third schema type; doesn't require the water fetch to proceed.

**Ready now (SO data available, C++-binary-only so far):**
- [x] Convert raw SO scrape files to native format and run through the built `MARVEL4` C++ binary — done 2026-08-21 (33S16O/34S16O/36S16O, single source), see dated section above.
- [x] Find the bridging transition (or confirm its absence in the literature) for the 2-component split in 33S16O/34S16O/36S16O — **done 2026-09-14, confirmed absent**. Akshay's combined-source run explicitly out of scope (not used going forward). `22HeStLy` (Heays et al. 2022, Mol. Phys. 121:17-18, arXiv:2301.05230) is the only published B3Sigma-/C3Pi electronic-band measurement of these three isotopologues — literature search found no independent second source to supply a bridging line. Treated as a genuine literature-connectivity gap (VO precedent, Bowesman et al. 2022: kept disconnected networks separate rather than force-merging synthetic transitions). 32S16O explicitly dropped from project scope (not a future task). See the dated section below for what actually shipped vs. what was tried and abandoned.
- [x] Wire SO into the Python port (`core/`) — done 2026-09-14, legacy (`core/`) path only, GNN deferred to a future session by Marco's choice. See dated section below for the unit bug, the QN-remap patch, and what's still unresolved.
- [x] Real 3-way C++/Python/oracle validation for SO — done 2026-09-17, refreshed 2026-09-18 after `MARVEL_scraping` fixed a real 22HeStLy F-label bug and reran their bootstrap (see 2026-09-18 dated section). `tests/test_so_oracle.py` now covers all three of 33S16O/34S16O/36S16O against the corrected data, ~1e-6/1e-7 cm⁻¹ (33/34S16O) and ~3e-2 cm⁻¹ (36S16O, one bridge-level reconstruction artifact, explained).
- [ ] Full SO GNN training (SO as a real training molecule, not just an eval-only check) — the eval-only OOD check was rerun 2026-09-18 against the corrected data (see dated section: GNN sigma correlates positively with real badness on all three isotopologues, r=0.63-0.78; legacy baseline correlates negatively on all three, r=-0.49 to -0.82, mechanism not yet understood). `MARVEL_scraping` separately offered 32S16O (real published Brady et al. 2024 ground truth) as an actual training candidate — a full 3-part plan for this (schema-encoding fix, training-loop wiring, 32S16O-drop conflict flagged) is written in the 2026-09-18 "Part 3" subsection, **not yet executed, awaiting explicit user go-ahead**.

**Blocked on data (Marco):**
- [ ] Fetch the water MARVEL input/output dataset into `C:\Code\MARVEL\molecules\water` (SO2 above may supersede this for schema validation, but water is still a separate target dataset).
- [ ] Optional: request the optimized-uncertainty CO input from the paper's authors to sharpen the uncertainty-head comparison.

**Deferred / revisit (out of scope for now — see "Explicitly out of scope"):**
- [ ] Wiring MARVEL_GNN into the parent repo's scraping pipeline (Step 7 reviewer agent) — kept decoupled on purpose; no interface designed.
- [ ] Isotopologue-mislabeling detection (distinct failure mode from QN label correction).
- [ ] Leaf-node / low-degree uncertainty strengthening within already-solved components.
