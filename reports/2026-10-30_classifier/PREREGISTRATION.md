# Step 1′: line classifier vs the iterative loop — pre-registration

**Written:** 2026-10-30, before any feature is computed or any model is fitted.
**Revised:** 2026-10-30, still before any computation, with vaultagent-26's addition: the label-free loop
L5 is also reported as a secondary baseline, with XGB − L5 and LR − L5 and their CIs. Previous version
sha256 d2cd0ca4…bb8a.
**Task:** vaultagent-26, approved by the user. Follows the db_MARVEL headroom study
(`reports/2026-10-30_dbmarvel_headroom/`, committed 849ad0c).
**Question:** can a line-level classifier on molecule-invariant features beat the iterative loop at ranking
real network negations, under leave-one-molecule-out? How much of the headroom does it recover, split into
residual-visible lines (|r|/unc ≥ 3) and residual-invisible ones? Uncertainty is out of scope: the headroom
study found no learnable target outside C2H4.

## Data (unchanged from the headroom study)

- **Runs:** the 21 outlier-usable runs in 13 molecule groups, on the same unedited graphs (largest
  component, original unc, negations flipped back in). Loaded by `dbmarvel_headroom.records()`.
  - Groups: CN, CaOH, AlO, VO (2 runs), HOCl, H3+, SO, TiO, PN, N2O, C2H4;
  - CO2 (628, 636, 626, 638, 727, 627); CO (12C-16O, 13C-16O).
- **Labels:**
  - positives: N (network negations);
  - X (whole-source exclusions; C2H4's 58AlPl): dropped from training and scoring;
  - negatives: every other line in the component.
- **Reused per-line quantities:** the headroom study's cached scores in `data/dbmarvel/scores/`: one-pass
  \|r\|/unc, studentized, leverage h (unc floored at 1e-6 cm-1), and the loop's removal order.

## Features (molecule-invariant; label-free)

**Leakage guard.** No feature uses:
- the negation labels (N, X, or which lines are negated);
- signed frequencies (only |ν| enters, as in the solve);
- `used_unc`, or anything derived from them.

Specific bans:
- The source's real negation rate.
- The headroom study's (iv) and (iv′) stop points. (iv) uses K = |N| + |X|, and (iv′)'s 3K cap binds in 10
  runs, so both depend on labels.

The loop feature below uses a new, label-free stop instead. In 1-column runs the single unc column is the
only one available. It may already be optimised, a threat carried over from the headroom study.

Per line, within its run's component:

| family | feature | definition |
|---|---|---|
| residual | `log_z` | log1p(\|r\|/unc), one-pass solve |
| residual | `log_stud` | log1p(studentized) |
| residual | `h` | leverage |
| residual | `loop_pos` | position in the **label-free loop L5**: drop the worst \|r\|/unc and re-solve the largest component, stopping at max \|r\|/unc ≤ 3 or after ceil(0.05 · component lines) removals. `loop_pos` = 1 − order / C for removed lines (C = L5's removal count), 0 for lines never removed (tied last). The L5 sequence is a prefix of the cached loop sequence, extended where the cache stops short. |
| structure | `log_deg_u`, `log_deg_l` | log1p(degree) of the upper and lower level in the component |
| source | `src_flagged` | share of the source's component lines with \|r\|/unc > 3 or removed by L5 |
| source | `src_scatter` | log10(1.4826 · median(\|r\|/unc) over the source's lines + 1e-3) |
| source | `src_size` | log10(source lines in the component) and source lines / component lines (2 features) |
| source | `unc_rel_src` | log10(unc / median unc of the source's lines) |
| duplicate/superseded | `has_dup` | another source has a line with identical upper and lower labels |
| duplicate/superseded | `n_dup_src` | log1p(number of other sources measuring that transition) |
| duplicate/superseded | `best_other_ratio` | log10(min other-source unc / this unc); 0 without a duplicate |
| duplicate/superseded | `dup_discrep` | log1p(\|ν − ν_best\| / sqrt(unc² + unc_best²)), against the most precise other-source line; 0 without one |
| duplicate/superseded | `newer_dup` | share of other-source duplicate lines whose source year is later; 0 without one |
| duplicate/superseded | `src_year_pct` | percentile of the source's year among the run's sources with a year; 0.5 if unknown, plus `year_missing` (0/1) |

- **Source year:** the leading two digits of the tag (yy ≥ 30 → 19yy, else 20yy). Tags without them
  (CDMS, CDSD, PGOPHER-…) have no year.
- **Source:** the tag without its `.n` counter.
- **Count:** 18 features in 4 families (residual 4, structure 2, source 5, duplicate/superseded 7).

## Models (leave-one-molecule-group-out, 13 outer folds)

**Training rows.**
- Lines of the 12 training groups: all positives, plus a seeded 20% sample of negatives per run (seed 0,
  weight × 5).
- Sample weights give each training group equal total weight, split equally over its runs.
- Training never sees the held-out group.

**Model 1, logistic regression (the floor).**
- L2-penalised and weighted, fitted with scipy L-BFGS.
- Features standardised on the training rows of each fit.
- Penalty λ ∈ {1e-3, 1e-2, 1e-1, 1}.

**Model 2, XGBoost** (CPU, `hist`; the version is recorded at install).
- Fixed: learning_rate 0.05, subsample 0.8, colsample_bytree 0.8, reg_lambda 10, seed 0.
- Grid: max_depth ∈ {2, 4} × n_estimators ∈ {100, 300} × min_child_weight ∈ {5, 50}.
- Monotone constraints, + on `log_z`, `log_stud`, `loop_pos` and `dup_discrep`; none on the other features.

**Model 2′, XGBoost without the duplicate/superseded family** (ablation). Same grid and procedure.

**Hyperparameter choice: nested.**
- Inside each outer fold, leave one of its 12 training groups out at a time.
- Pick the setting with the best mean inner AP_ties (each inner group weighted 1). Refit on all 12 groups.
- Predict the held-out group.
- The held-out molecule is never used for tuning.

**Baselines (unchanged, taken from the headroom study's results):**
- the loop (iv′) — the primary bar;
- the loop (iv), and one-pass \|r\|/unc — secondary;
- **the label-free loop L5** — secondary, scored as −removal order with never-removed lines tied last. L5 is
  the loop that would actually be deployed. (iv′) stays the primary bar, the conservative choice, because
  its label-derived cap flatters it.

## Metrics

**AP.** All AP is tie-aware (`ap_ties`, the headroom study's corrected metric), on N against negatives per
run.
- Pooling: each group weighted 1, split over its runs.
- CIs: 95% cluster bootstrap over the 13 groups (1000 resamples, seed 0), on paired per-group differences.

**Primary.**
- Pooled AP of XGB − AP of loop (iv′), with CI.
- Pooled AP of LR − AP of loop (iv′), with CI.

**Secondary:**
0. Pooled AP of XGB − AP of L5, and of LR − AP of L5, each with CI.
1. **Headroom recovery.**
   - The headroom study's H lines (iv′ partition primary, iv secondary).
   - Measure: the share each model ranks in the top K of its run (K = |N|).
   - Split by one-pass |r|/unc ≥ 3 (visible) vs < 3 (invisible). Pooled, molecule-weighted.
   - Reported against the random expectation K / pool.
2. **Negate vs inflate.** In 2-column runs (VO, H3+, CO2, PN, CO 12C-16O, N2O; 6 groups), among lines with
   |r|/unc ≥ 3:
   - positives N; negatives: raised-uncertainty lines (used > original); other lines left out;
   - AUROC per run and pooled, for XGB, LR, |r|/unc and `loop_pos`.
   - `used_unc` appears only as this label, never as a feature.
3. **Per-group AP** for every model and baseline.
4. **Feature importance.**
   - XGB total gain, averaged over outer folds.
   - Permutation importance on the held-out predictions: each feature, and each family as a block, permuted
     within each held-out run (3 repeats, seed 0). Measure: drop in pooled AP.
5. **Duplicate/superseded contribution** (as asked):
   - AP(XGB) − AP(XGB without dup), with CI;
   - invisible-H recovery with vs without the family;
   - that family's permutation importance.

## Decision rule, stated in advance

The rule is judged on (iv′) only. The L5 comparisons inform the reading; they don't change the rule.

**Escalate** to any GNN or loop+GNN hybrid only if both hold:
- **(a)** pooled AP(XGB) − AP(loop iv′) > 0 with the 95% CI excluding 0; **and**
- **(b)** XGB ranks ≥ 0.25 of the headroom-study H lines (iv′ partition, molecule-weighted) within the top
  K. That is a quarter of the headroom, about 7% of all negations. It must also be at least twice the random
  expectation.

**Refinements:**
- If (a) and (b) hold, but the duplicate/superseded family carries most of the gain, the next step is a
  tabular loop + classifier hybrid, not a GNN. "Most" means AP(XGB) − AP(XGB without dup) ≥ half of
  AP(XGB) − AP(loop). A GNN adds graph context, not source metadata.
- If (a) holds but not (b): the learned ranking helps the visible lines only. No escalation; report a
  possible tabular re-ranker.
- If (a) fails: no escalation. The loop stays the tool for outliers.

## Threats to validity, stated in advance

- **13 groups is a small outer sample.** The CIs will be wide, and CO2 is a single group of 6 runs.
- **PN is atypical** (52% negated, original unc far too small). It weighs 1/13. Per-group AP is reported.
- **The loop baselines differ in label use.** (iv′) still carries its label-derived 3K cap in 10 runs, which
  flatters the bar the models must beat. The models' own loop feature (L5) is label-free.
- **Source-year parsing is heuristic.** Unparseable tags get the missing flag.
- **1-column runs** may carry optimised uncertainties, as in the headroom study.
