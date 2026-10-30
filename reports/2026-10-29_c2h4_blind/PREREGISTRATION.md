# Blind C2H4 test — pre-registration

**Written:** 2026-10-29, before the answer key is released (overrides.tsv, decisions.md not opened).
**Revised:** 2026-10-29, folding in vaultagent-f7's four step-B changes (primary endpoints; (iv) in the
AP CI with ties at the bottom; split uncertainty label; disagreement reporting). Still before the key.
**Question:** in one pass, does a CO-trained GNN find the lines that 12C2H4 v1 curation negated, and does it
raise uncertainty where curation raised it? Heads under test: outlier and uncertainty. Label-correction and
orphan-linkage heads are out of scope.
**Blindness is procedural.** The edited trans file (negation signs, uncertainties) has been seen. Three
safeguards apply:
- no C2H4 data is used for training or tuning;
- this document is fixed before the key is released;
- predictions are frozen by sha256 before scoring.

## Model

- Trained on CO only, with the 2026-08-24 split, hyperparameters and seeds (`TRAIN_ISOS = 13C16O, 12C18O,
  13C17O`; `VAL_ISOS = 12C17O, 13C18O`; 1000 epochs; checkpoint chosen by best validation uncertainty NLL).
- **QN-masked** (`QN_MASK = True` in `gnn/data.py`): the QN token block is all zeros and the token-count
  column is constant, in training and in inference alike. `MAX_QN = 15` only so nothing is truncated.
  The heads therefore see network structure, uncertainties, frequencies, residuals and leverages,
  never labels.
- Checkpoint `data/marvel_gnn.pt`, sha256 `9440b7a011fcbd1277b1d0539c42184eb9acf724d25e3003b68282c6fbd9b73f`.
  Best epoch 450; log `data/train_v8_qnmask.log`.
- CO held-out results, 12C17O / 13C18O, against the 2026-08-24 unmasked model:

  | metric | masked | 2026-08-24 |
  |---|---|---|
  | uncertainty NLL | 7.14 / 6.79 | 7.19 / 6.44 |
  | outlier AP | 0.206 / 0.205 | 0.210 / 0.167 |
  | freq-shift AP | 0.040 / 0.027 | 0.041 / 0.027 |

  Random AP is 0.039 / 0.049.

## Data (frozen at MARVEL_network 0b56238)

- **Unedited graph:** the 47 per-source inputs (`git archive 0b56238 molecules/C2H4/input`), pooled:
  50,611 lines.
  - All 91 source-file negations are flipped back to positive as ordinary lines. The v1 report's
    figure of 81 excludes 58AlPl's 10.
  - 58AlPl stays in, even though curation excluded that whole source (54 lines).
  - Blank per-line uncertainties take `stated_unc` from MARVEL_network HEAD `sources.tsv` (last changed in
    commit df0978a; sha256 1eea6435…94efc; the file isn't in 0b56238), else a fixed 1e-3 cm-1:

    | source | rule | unc (cm-1) |
    |---|---|---|
    | 10LoBoLoRo | stated | 0.0005 |
    | 24PeLeSuRu | stated | 0.005 |
    | 26PeLeSuRu | stated | 0.0013 |
    | 18AsRa | none stated, fixed | 1e-3 |
    | 23MrMaReAr | none stated, fixed | 1e-3 |
    | 24BeBeReMa | none stated, fixed | 1e-3 |

  - unc = 0 takes the parser floor, 1e-6 cm-1.
- **Edited graph:** the v1 deliverable trans with negated lines dropped: 49,926 lines.
- Both graphs use only the **ground-state component** (the one containing the all-zero level, asserted
  to be the largest):

  | graph | levels | lines |
  |---|---|---|
  | unedited | 11,687 | 50,294 |
  | edited | 11,580 | 49,615 |

  Lines outside it get `in_main = 0`. They are counted per category but not scored.
- **Line identity:** the input `id` column, which equals the deliverable `ref`. The join is exact: every
  deliverable id is found in the inputs, and the only ids missing from the deliverable are 58AlPl's 54.

## Predictions (one pass each, written to `data/C2H4_v1/predictions/`)

Per line (`*_lines.tsv`):

| column | definition |
|---|---|
| `resid_over_unc` | \|ν − (E_u − E_l)\| / unc, from the Python-port weighted least-squares solve of that graph |
| `studentized` | resid / (unc · sqrt(1 − h)), with h from sparse `edge_leverages` |
| `gnn_outlier_logit` | the GNN outlier head's logit |
| `gnn_line_sigma` | sqrt(σ_u² + σ_l²) in cm-1: the GNN's per-level uncertainty σ (uncertainty head) mapped to the line |
| `gnn_sigma_over_unc` | `gnn_line_sigma` / unc |
| `random` | uniform draw, seed 20261029 |
| `iterative_rank` | unedited graph only; see baseline (iv) |

Per level (`*_levels.tsv`): energy and GNN σ (cm-1).

## Primary endpoints

Everything not listed here is secondary.

1. **Outlier.** AP on pooled N (unedited graph), GNN vs the best of baselines (i), (ii) and (iv), with
   the 95% bootstrap CI on AP(GNN) − AP(best).
2. **Uncertainty.** AUROC of `gnn_sigma_over_unc` for line-level `set_unc` (definition (a) below) on the
   unedited graph, with a 95% bootstrap CI (1000 resamples, seed 0), against the same baselines.

## Outlier metrics (unedited graph)

**Sets**, defined from the released key, with lines matched by id:
- **N** (network negations): lines positive in the inputs but negated in the v1 deliverable (550). Scored
  pooled, and per category: QAB, QAC, QAI/QAM, QAJ, QAU. A category's members are the N lines the key
  assigns to it. A line in several categories counts in each.
- **S** (source negations): the 91 flipped lines, scored per source as well as pooled.
- **X** (excluded sources): lines of sources that curation dropped wholesale (58AlPl, 54). Scored as their
  own category.
- **Negatives:** lines in the ground component that are in none of N, S or X.

**For each positive set P** (N pooled, each N category, S, each S source with at least 1 line in the
ground component, X), the pool is P ∪ negatives, restricted to the ground component, with the other
positive sets left out. On that pool:
- **AP** (average precision, ties broken by line order);
- **recall@k**, with k = |P in the pool|;
- **recall in the top 1%** of the pool, i.e. the ceil(0.01 · pool size) highest-scored lines.

**Scores compared** (higher = more suspicious):
1. GNN `gnn_outlier_logit`.
2. Baseline (i) `resid_over_unc`: MARVEL's own one-pass flag.
3. Baseline (ii) `studentized`: sparse leverage, no GNN.
4. Baseline (iii) random: the seeded draw, plus the analytic expectation (AP = prevalence,
   recall@k = k/pool, top-1% recall = 0.01).
5. Baseline (iv), **included**: a simulated iterative MARVEL loop.
   - Repeat 641 times (= 550 + 91): drop the single worst `resid_over_unc` line and re-solve the
     ground component.
   - Score = −removal order.
   - Lines never removed are tied at the bottom: one shared score for all of them, including lines
     cut off from the ground component by a removal. (The prediction file still records their
     final-residual order in `iterative_rank`; scoring ignores it.)
   - (iv) is not one-pass; it is the loop the GNN would replace.

**Uncertainty:** 95% bootstrap intervals (1000 resamples of the pool, seed 0) on AP, and on
AP(GNN) − AP(best baseline among (i), (ii) and (iv)), for N pooled and S. "Best" is chosen on the full pool,
then held fixed across resamples. For (iv), the lines never removed are scored as tied at the bottom
(a single equal score), not ordered by final residual.

**Interpretation, stated in advance:**
- Expected outlier-shaped categories: QAJ, QAU, QAC. Expected rule-based categories: QAB, QAI/QAM.
- Lines with leverage h ≈ 1 (bridges) are undetectable from residuals by construction. Recall is also
  reported separately for positives with h ≥ 0.99.

## Uncertainty metrics

**Unedited graph: did the GNN raise uncertainty where curation did?** The label is split in two, so
that the roughly 12k lines from scaled sources don't swamp the roughly 836 line-level decisions.

- **(a) Line-level.**
  - Positives: lines with a `set_unc` row in the key.
  - Negatives: ground-component lines outside N, S and X, without a `set_unc` row, and not in a scaled
    source.
  - Scaled-source lines are excluded from both sides unless they have their own `set_unc` row.
  - Results are stratified by key category. The K-doublet half-separation rule (QAS a′) is reported
    separately as rule-based, like QAB.
- **(b) Source-level.** For every source, the median `gnn_sigma_over_unc` over its eligible lines is set
  against its applied scale factor, or its assumed value for the 6 blank-unc sources. Every source is
  reported. Spearman ρ is given only if the key has at least 5 scaled sources; otherwise the result
  is descriptive.
- **Scores:** primary `gnn_sigma_over_unc`, secondary `gnn_line_sigma`. Baselines: `resid_over_unc`,
  `studentized`, random.
- **Metrics for (a):** AUROC; Spearman ρ of each score against the applied factor among the positives.
- **Applied factor vs key:** applied factor = deliverable optimized unc / unedited unc. Where it
  disagrees with the key (a `set_unc` row with factor 1, or a factor ≠ 1 with no key row or scale),
  report counts and examples. No disagreeing line is dropped silently.

**Edited graph: GNN level σ vs the v1 bootstrap uncertainties in `states.txt`.**
- Spearman ρ over ground-component levels, excluding the ground level. Reported overall and per
  `states.txt` status (validated / tree).
- Disagreements: the 20 levels with the largest log10(σ_GNN / unc_states) in each direction, with their
  degree and status.
- Interpretation caveat: the GNN σ was trained on CO masked-refit errors, not on MARVEL bootstrap
  uncertainties, so a perfect correlation is not expected.

## Advisory output

The top **N = 50** lines of the edited graph by `gnn_outlier_logit` go to marvel-network-eb for later
review, with `resid_over_unc`, `leverage` and `studentized`. Nothing goes to MARVEL_network.

## Threats to validity, stated in advance

- **Domain shift.** The model was trained on CO diatomic networks of 33–863 levels; C2H4 has 11.6k levels,
  4 ground-state symmetry blocks joined by bridges, and 46 sources.
  - CO uncertainties come from one compilation. C2H4's span 3e-7 to 0.3 cm-1.
  - Masking the QN block removes label shift but not this shift.
- **Training distribution.** The outlier head was trained on synthetic 5% corruptions: frequency shifts of
  5–500 unc, and J±1 bumps. Curation negates for other reasons too (rule-based categories), which
  structure can't see.
- **Unedited uncertainties include placeholders** for 6 sources (table above). The fixed 1e-3 cm-1 values
  bias `resid_over_unc` for those sources' lines.
- **No CO seed-variance estimate exists** (the 2026-10-15 skeptic review's seed table is still pending).
  Step A's CO comparison with 2026-08-24 is a single run.

- **Prior from CO (stated before scoring):**
  - The outlier head detects topology rewiring (QN-bump AP about 0.28), not frequency errors on intact
    topology: freq-shift AP is about 0.03, roughly random.
  - On CO, raw |resid|/unc already beats the GNN's outlier AP (2026-10-15 skeptic review: raw 0.230 /
    0.269 vs GNN 0.210 / 0.167).
  - Expected outcome: baselines (i) and (ii) are hard to beat on frequency-shaped negations. The GNN has
    its best chance on misassignment-shaped lines.

## Changes after this point

Any analysis not listed here is labelled **post hoc** in the report.
