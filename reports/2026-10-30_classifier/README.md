# Step 1′: line classifier vs the iterative loop

**Task:** vaultagent-26, approved by the user. **Status:** done; negative.
**Pre-registration:** `PREREGISTRATION.md`, sha256
`d20e6fec1113c3205c4dba7fc51c04309d3a622708d3429eb350a6bdf0a0b521`. This is the revised version: it adds the
label-free loop L5 as a secondary baseline and was written before any computation.
**Files:**
- script: `scripts/classifier_step1.py`;
- outputs: `results.json`, `results.txt`;
- caches: the L5 loop orders in `data/dbmarvel/scores/*_L5.npz`; the run log is
  `data/dbmarvel/classifier_run.log`.
**Runtime:** about 25 min on CPU. xgboost 3.4.1.

## Question

Can a line-level classifier on molecule-invariant features beat the iterative loop at ranking real network
negations, under leave-one-molecule-out? How much of the headroom study's 27% headroom does it recover?

## Deviation from the pre-registration

- **XGBoost is fitted with the native `xgb.train` / `Booster` API**, not the `XGBClassifier` wrapper. The
  wrapper needs scikit-learn, which is not an approved dependency.
- The model, parameters, grid and monotone constraints are as pre-registered. Tree count = `num_boost_round`.
- Nothing else changed.

## Primary result (decision rule judged on (iv′))

Pooled AP, molecule-weighted and tie-aware, with 95% cluster-bootstrap CIs over 13 groups.

| score | AP | 95% CI |
|---|---|---|
| loop (iv), K from the labels | 0.595 | [0.496, 0.707] |
| logistic regression | 0.560 | [0.493, 0.641] |
| **loop (iv′), primary bar** | **0.544** | [0.446, 0.645] |
| loop L5, label-free (what would be deployed) | 0.540 | [0.443, 0.645] |
| XGBoost | 0.499 | [0.405, 0.613] |
| XGBoost without the dup/superseded family | 0.468 | [0.371, 0.589] |
| one-pass \|r\|/unc | 0.407 | [0.299, 0.529] |

Paired differences:

| difference | mean | 95% CI |
|---|---|---|
| **XGB − loop (iv′)** | **−0.045** | [−0.185, +0.094] |
| LR − loop (iv′) | +0.016 | [−0.062, +0.111] |
| XGB − L5 | −0.041 | [−0.181, +0.096] |
| LR − L5 | +0.020 | [−0.059, +0.113] |
| XGB − XGB without dup | +0.031 | [−0.019, +0.085] |

**Decision rule:**
- (a) fails: XGB is below the loop, and the CI spans 0.
- (b) fails: XGB ranks 0.126 of the H lines in the top K, against the 0.25 bar (random is 0.042).
- **Verdict (pre-registered): no escalation. The loop stays the tool for outliers.**
- The logistic regression floor ties the loop. Neither the conservative bar (iv′) nor the deployable loop L5
  is beaten.

**Per-group XGB − (iv′):**
- gains on SO (+0.30), TiO (+0.25) and AlO (+0.24), the three groups whose label-free loop stops early;
- losses on CO (−0.71; N = 12 and 15), VO (−0.38) and H3+ (−0.18);
- within ±0.05 elsewhere.

LR shows the same pattern, with smaller losses: CO −0.25, H3+ −0.31; HOCl +0.21.

## Headroom recovery (secondary)

Share of the headroom study's H lines (iv′ partition) ranked within the top K = |N|. Pooled,
molecule-weighted.

| | all H | visible (\|r\|/unc ≥ 3) | invisible (\|r\|/unc < 3) |
|---|---|---|---|
| XGB | 0.126 | 0.155 | 0.140 |
| XGB without dup | 0.118 | 0.191 | 0.125 |
| LR | 0.179 | 0.368 | 0.065 |
| one-pass \|r\|/unc | 0.194 | **0.515** | 0.037 |
| random | 0.042 | 0.050 | 0.049 |

- **Visible headroom.** The one-pass residual already puts half of the visible H lines in its top K. The loop
  misses them for one of two reasons (not separated here):
  - its label-free stop comes early: AlO, TiO and SO stop at 627, 535 and 149 removals, against K = 1499,
    1416 and 388;
  - re-solving after a removal lowers their residual.
  - So part of the visible headroom is recoverable without learning, by mixing one-pass and loop ranks.
  - That is not tested here; it is a cheap deterministic follow-up.
- **Invisible headroom.** XGB reaches 0.14, about 3× random. The dup family adds only 0.015 of that. Most of
  the invisible headroom stays invisible.

## Duplicate/superseded features (reported separately, as asked)

- **AP:** XGB − XGB without dup = +0.031 [−0.019, +0.085]. The CI includes 0. Gains on N2O (+0.24), CO2
  (+0.13) and H3+ (+0.11); losses on VO (−0.14) and AlO (−0.08).
- **Permutation importance:** the family as a block costs 0.098 pooled AP when permuted. Almost all of that
  is `dup_discrep` (0.084), the duplicate's disagreement over the combined uncertainty. That signal is
  residual-like. Year and precision ratios do little (`src_year_pct` 0.022; `newer_dup`,
  `best_other_ratio` ≈ 0).
- **Invisible-H recovery:** 0.140 with the family, 0.125 without.
- **Reading:** "superseded by a newer or better measurement" is not what explains the residual-invisible
  negations, at least as encoded here.

## What the model learns (secondary)

**Permutation drop in pooled AP:**
- residual family 0.43 (`loop_pos` 0.29, `log_z` 0.16, `log_stud` 0.04);
- dup 0.10;
- source 0.01;
- structure 0.01.

**XGB gain share** puts `src_scatter` first (0.26), then `loop_pos` 0.13 and `src_flagged` 0.12. But permuting
the source features costs nothing on held-out molecules. They fit the training molecules and don't transfer.
That is the likely reason XGB lands below LR.

**Overall:** the classifier mostly re-learns the loop's own ranking.

## Negate vs inflate (secondary)

AUROC among |r|/unc ≥ 3 lines in 2-column runs: positives are network negations, negatives are lines whose
uncertainty was raised. 6 groups.

| score | AUROC | 95% CI |
|---|---|---|
| L5 loop position | 0.851 | [0.70, 0.95] |
| LR | 0.824 | |
| XGB | 0.789 | |
| \|r\|/unc | 0.781 | |

- PN sits at about 0.45 for every score.
- **Interpretation:** the "raised" side is mostly MARVEL 4's automatic uncertainty optimisation, not a
  curator choice (headroom study). So this measures how well a score separates curator negations from lines
  the algorithm inflated afterwards, not a curator's negate-vs-inflate judgement.
- On that question the label-free loop is already the best of the four.

## Answer

- **No.** Under leave-one-molecule-out, neither model beats the loop.
  - XGB is −0.045 [−0.19, +0.09] against (iv′) and −0.041 against the deployable L5.
  - LR is +0.016 [−0.06, +0.11] and +0.020: a statistical tie.
- **Headroom recovered:** XGB 13% of H (visible 16%, invisible 14%); LR 18% (37% / 7%). Random is about 4–5%.
- **Pre-registered decision:** no GNN and no loop+GNN hybrid.

**Recommendations (mine):**
1. **Visible headroom:** half of it sits in the one-pass residual's top K. The cheap deterministic follow-up
   is a loop that doesn't let re-solving hide a line that was flagged in one pass. For example, rank by the
   max of the one-pass and loop positions. It needs no learning. Untested.
2. **Invisible headroom:** none of these features finds it, including duplicate/superseded information. It
   most likely reflects evidence outside the line list (literature, assignments, source reliability). It
   stays a job for human curation.

## Threats to validity

- **13 outer groups.** The CIs are wide: a true gain of up to about +0.1 AP cannot be excluded for either
  model.
- **CO's loss is large but rests on 27 positives.** Dropping CO would not flip (a) for XGB: the other 12
  groups' mean difference is still ≤ about +0.01.
- **(iv′) is flattered by its label-derived cap.** The label-free L5 scores almost the same (0.540 vs 0.544),
  so the conclusion doesn't depend on it.
