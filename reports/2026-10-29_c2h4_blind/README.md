# Blind C2H4 test: CO-trained GNN vs 12C2H4 v1 network curation

**Date:** 2026-10-29
**Requested by:** vaultagent-f7, with the user's approval; the user approved the retrain in this session.
**Heads tested:** outlier and uncertainty.
**Status:** advisory only. Nothing goes to MARVEL_network.

Files in this folder:
- `PREREGISTRATION.md`: sha256 `2e0c5149…fa18`. Frozen before the key was released, and its hash was
  checked independently by vaultagent-f7.
- `score_output.txt`: the full pre-registered output.
- `posthoc_output.txt`: post hoc controls.
- `results.json`.
- `advisory_top50_edited.tsv`.

## Verdict

**The CO-trained GNN does not replace any part of the negate-and-rerun loop on C2H4.**

- **Outlier:** the GNN loses to every non-random baseline. It ranks lines almost like one-pass |resid|/unc
  (ρ = 0.78) and does slightly worse. The simulated iterative loop beats all one-pass scores by about 8x
  in AP.
- **Uncertainty:** the pre-registered primary endpoint nominally beats the residual baselines. A post hoc
  control shows that the edge comes entirely from 1/unc. The GNN's level σ has collapsed to an almost
  constant value on C2H4 (6.9–8.8e-4 cm-1 for every level).

## Model and data (pre-registered)

**Model.**
- Trained on CO with the QN block masked (`QN_MASK = True`) and `MAX_QN = 15`. Split, hyperparameters and
  seeds as on 2026-08-24.
- Checkpoint `data/marvel_gnn.pt`, sha256 `9440b7a0…b73f`.

CO held-out results, 12C17O / 13C18O:

| metric | masked model | 2026-08-24 model |
|---|---|---|
| uncertainty NLL | 7.14 / 6.79 | 7.19 / 6.44 |
| outlier AP | 0.206 / 0.205 | 0.210 / 0.167 |

Masking the labels cost nothing on CO.

**Graphs, frozen at MARVEL_network 0b56238; ground-state component only:**

| graph | levels | lines | built from |
|---|---|---|---|
| unedited | 11,687 | 50,294 | pooled inputs, with the 91 source negations flipped back in |
| edited | 11,580 | 49,615 | v1 deliverable, negated lines dropped |

**Predictions:** frozen by sha256 (`data/C2H4_v1/predictions/SHA256SUMS`) before the key was released. The
scorer re-checks every hash.

## Key mapping (overrides.tsv at 0b56238)

The deliverable's `C2H4_12C2H4_overrides.tsv` is byte-identical to the 0b56238 file (sha256 2f0d9025…).

| row type | rows | pre-registered set |
|---|---|---|
| `negate` | 550 | **N**. Categories from the reason tag: QAC 350; QAB 75 (67 + 8 tagged QAB/QAI); QAI/QAM 65 (57 + 8); QAJ 63; QAU 5. Every row has a category, and every target is found in the unedited lines. 544 are in the ground component. |
| `set_unc` | 827 | Line-level uncertainty positives (a): QAS a′ 500 (rule-based), QAS 08RoBoVa 33, QAR 286, QAU 17. Some lines carry 2 tags. All 827 are eligible lines. |
| `scale` | 5 | Source-level (b): 02MoKoBuHe ×3, 83DaPiFaDe ×5, 19UlBeGrRa ×2, 24BeVaTuBo ×2, 25BeBeReCa ×2. |
| `assume_unc` | 6 | Source-level (b): factor = assumed value / pre-registered placeholder. |
| `exclude` | 2 | **X** = 58AlPl (54 lines). 10UlOnBePe is **not in the 0b56238 inputs at all**, so this row fits no set and is not scored. |

**S** (source negations) = the 91 flipped lines, all in the ground component.

**Applied factor vs key:**
- 0 `set_unc` rows with factor ≈ 1;
- 0 lines with factor ≠ 1 and no `set_unc`, `scale` or `assume_unc`;
- 0 scaled-source lines whose factor differs from the scale.

The key and the deliverable agree exactly.

## Primary endpoint 1: outlier AP on pooled N (544 positives, pool 50,159)

| score | AP [95% CI] | recall@k | top 1% |
|---|---|---|---|
| GNN logit | 0.050 [0.041, 0.060] | 0.040 | 0.037 |
| (i) \|resid\|/unc | 0.054 [0.044, 0.066] | 0.048 | 0.042 |
| (ii) studentized | 0.052 [0.042, 0.064] | 0.044 | 0.042 |
| **(iv) iterative loop** | **0.425 [0.381, 0.478]** | **0.467** | **0.465** |
| (iii) random | 0.011 (expected 0.0108) | 0.011 | 0.011 |

- The best baseline is (iv). AP(GNN) − AP(iv) = **−0.38, 95% CI [−0.426, −0.334]**.
- Against the one-pass baselines: GNN − (i) is [−0.007, −0.002] and GNN − (ii) is [−0.005, −0.001].
  The GNN loses narrowly but significantly to both.

### Per category (secondary)

| category | n | GNN AP | (i) | (ii) | (iv) | random |
|---|---|---|---|---|---|---|
| QAC (fixed-GS line test) | 350 | 0.045 | 0.050 | 0.049 | **0.518** | 0.007 |
| QAJ (batch 2, QL) | 63 | 0.003 | 0.004 | 0.004 | **0.249** | 0.001 |
| QAU (batch 3) | 5 | 0.000 | 0.001 | 0.001 | **0.181** | 0.000 |
| QAB (GS-block joins, rule) | 75 | 0.010 | 0.012 | 0.010 | 0.018 | 0.002 |
| QAI/QAM (selection rule) | 59 | 0.001 | 0.001 | 0.001 | 0.006 | 0.002 |

- This matches the stated expectation. The rule-based categories (QAB, QAI/QAM) are invisible to every
  score. Of the 59 QAI/QAM lines, 37 are bridges (h ≥ 0.99).
- The outlier-shaped categories (QAC, QAJ, QAU) are found only by the iterative loop.
- Bridge positives (45 with h ≥ 0.99): recall@k is 0 for every score, as predicted.

### Source negations S (91; secondary)

| score | AP [95% CI] | recall@k | top 1% |
|---|---|---|---|
| GNN | 0.021 [0.011, 0.040] | 0.055 | 0.099 |
| (i) \|resid\|/unc | 0.023 | 0.055 | 0.132 |
| (ii) studentized | 0.022 | 0.066 | 0.121 |
| (iv) iterative | 0.279 [0.174, 0.391] | 0.407 | 0.484 |

- GNN − (iv) = [−0.365, −0.157].
- Per source: 24BeVaTuBo (25 lines) is found by (iv) with AP 0.458, against 0.007 for every one-pass
  score.
- 97SaOoReFa's 18 term-value lines are found by nothing.

**X (58AlPl, 54 lines):** AP is about 0.001 for every score. The source is consistent with the network;
curation excluded it for coverage and connectivity reasons, not as outliers.

## Primary endpoint 2: uncertainty AUROC, line-level set_unc (827 positives vs 36,723 negatives)

| score | AUROC [95% CI] |
|---|---|
| GNN σ_line / unc | **0.739 [0.728, 0.750]** |
| GNN σ_line (raw) | 0.337 [0.320, 0.354] |
| (i) \|resid\|/unc | 0.700 [0.683, 0.717] |
| (ii) studentized | 0.711 [0.694, 0.726] |
| random | 0.507 |

As pre-registered, GNN − (i) = [+0.020, +0.057] and GNN − (ii) = [+0.011, +0.047]. The GNN wins nominally.

**POST HOC, not pre-registered. It decides the interpretation:**
- The GNN's level σ is almost constant on C2H4: 6.85e-4 to 8.77e-4 cm-1, p5–p95 7.0–8.4e-4, on both
  graphs. So σ_line/unc is essentially 1/unc: Spearman ρ = 0.925 between them over the pool.
- 1/unc alone gives AUROC **0.755**. GNN − 1/unc = [−0.019, −0.013].
- Curation raised uncertainty on lines that had small stated uncertainties. The GNN adds no information
  beyond that.
- The raw GNN σ scores *below* random (0.337). Its small variation runs opposite to curation.

Per category, GNN σ/unc vs (i): QAS a′ (rule) 0.765 vs 0.752; QAS 08RoBoVa 0.958 vs 0.849; QAR 0.669 vs
0.588; QAU 0.710 vs 0.877. All carry the same 1/unc caveat.

Spearman of each score against the applied factor among the positives: GNN σ/unc −0.10, (i) +0.14,
(ii) +0.18.

### Source level (b)

- Spearman over the 5 scaled sources (factor vs median GNN σ/unc): +0.45, on 5 points.
- Descriptively, the GNN's median σ/unc mostly reflects each source's stated uncertainty. Examples: 98RuFiKhHe
  (sub-Doppler) 1.5e3, Magic 1.2e3, 59DoSt (0.03 cm-1) 0.04.
- The assume_unc factors (0.2–2) don't line up with it. Full table in `score_output.txt`.

### Edited graph: GNN σ vs states.txt bootstrap uncertainties (secondary)

- Spearman +0.50 overall: validated +0.54, tree +0.15.
- **POST HOC:** the minimum incident line uncertainty alone gives +0.52, and 1/degree gives +0.36.
  The GNN's correlation is no better than that trivial structural feature.
- Disagreements are driven by GNN σ being near-constant:
  - GNN >> states: low-J ground and v12 levels, where states.txt says 1e-6 to 1.7e-4 and the GNN says
    about 8e-4.
  - GNN << states: v7+v8 tree levels, where states.txt says 0.01–0.02 and the GNN says about 7e-4.

## Advisory output (top 50 edited-graph lines by GNN logit)

`advisory_top50_edited.tsv`.

- **All 50 are bridges (h = 1) with zero residual** (13UlGrAsHo, etc.).
- No line in the edited graph reaches p > 0.5; the maximum logit is −0.57.
- The GNN's top flags are therefore "unverifiable from network structure", not "likely wrong". On CO the
  head learned that dangling and rewired topology signals a QN bump.
- **Decision (vaultagent-f7):** the list is held, not forwarded to marvel-network-eb. A list of bridges
  adds nothing to MARVEL's own tree flag.

## Where the GNN wins and loses

- **Loses:** to iterative re-solve on every outlier set (−0.38 AP on N, −0.25 on S); narrowly to one-pass
  |resid|/unc and studentized on N; to 1/unc on uncertainty (post hoc).
- **Ties:** with everything on the rule-based categories (QAB, QAI/QAM) and on X. All scores are
  near random there, as expected.
- **Wins:** only nominally, on the pre-registered uncertainty AUROC against the residual baselines. The
  post hoc 1/unc control shows this is not a GNN capability.

## Why (interpretation, post hoc)

1. **Domain shift.** The uncertainty head outputs about 7.5e-4 cm-1 for every C2H4 level, i.e. it has
   collapsed to something like a prior mean. The likely cause is that C2H4's network size and
   uncertainty mix fall outside what the head saw on CO. I haven't checked which feature drives it.
2. **One-pass vs iterative.** The real negations (QAC, QAJ) are found by the loop because removing one bad
   line un-masks its neighbours. A single pass, by GNN or residual, sees the bad lines' residuals
   smeared around their cycles. The CO feature probe found the same: the studentized residual localises
   the cycle, not the culprit.
3. **The outlier head flags topology, not frequency errors.** CO freq-shift AP was about random (stated in
   the pre-registration). On C2H4 its top flags are bridges.

## Implications

- Don't use the current GNN heads for C2H4 or similar large polyatomic networks.
- The useful baseline is cheap: a residual-driven iterative negate-and-re-solve on the unedited input
  recovers 47% of curation's network negations in its first 641 removals, with no model at all.
- If the GNN direction continues:
  - train on corruption mixes that resemble real negations (small frequency shifts on redundant
    edges, at realistic uncertainty spreads);
  - include large polyatomic networks in training (CO2 is now unblocked on leverage);
  - consider iterative GNN inference, re-scoring after each removal.
- None of these are tested here.

## Carried-over items

- The CO2 oracle tests skip on the stale `C:\Code\MARVEL\molecules\CO2` path.
- No seed-variance estimate exists for CO training.
- The label-correction and orphan-linkage heads weren't tested. Masking weakens orphan linkage on CO
  (MRR 0.081 / 0.006 vs 0.128 / 0.052).
