# db_MARVEL headroom study — phase 2 pre-registration

**Revised:** 2026-10-30, before any phase 2 score is computed, with the user's change (via vaultagent-26):
the B/L/R/H partition is reported twice, with L from (iv) and with L from the label-free loop (iv′); the
step 1 thresholds apply to both. Previous version sha256 05cefb8f…3b49.
**Written:** 2026-10-30, after the phase 1 label audit (`audit.tsv`, README.md) and before any phase 2
score is computed. The audit looked at label counts and at the published solve only. No residual, leverage,
loop or rule result has been computed against the negation or uncertainty labels.
**Task:** vaultagent-26, approved by the user. No GNN training or inference.
**Question:** on real curated MARVEL datasets, how much of the curators' outlier and uncertainty work does
the no-model pipeline already recover? The no-model pipeline is the iterative loop plus cheap deterministic
checks. Whatever is left (not caught, not a bridge) is the room a learned model could have.

## Data

- **Source:** the read-only copy in `data/dbmarvel/` (db_MARVEL 010b888, clean tree; `SHA256SUMS`), parsed
  with db_MARVEL's own `loader/parse.py` and its manifests. Loader: `scripts/dbmarvel_audit.py` (`load`).
- **Runs:** those marked usable in `audit.tsv`, plus 12C2H4 v1 from `data/C2H4_v1/` (key at
  MARVEL_network 0b56238), built as in the blind test. Thresholds are fixed from label counts only:
  - outlier: at least 10 network negations in the largest component;
  - uncertainty: original and used uncertainty both present, at least 10 raised lines, and the published
    levels reproduced (at least 90% within their published unc).
- **Outlier runs (21 runs, 13 molecule groups):**
  - CN; CaOH; AlO; VO (fine and hfs); HOCl; H3+;
  - SO; TiO; PN; N2O (14N2-16O); C2H4;
  - CO2: 628, 636, 626, 638, 727, 627;
  - CO: 12C-16O (25MaElAb) and 13C-16O (26GrDaPo).
- **Uncertainty runs (20 runs, 7 molecule groups):**
  - VO (2); H3+ family (H3+, H2D+); PN; CO (12C-16O); N2O; C2H4 (key);
  - CO2 (12): 628, 828, 728, 838, 636, 626, 737, 738, 638, 637, 727, 627.
- **Not usable:** H2O 09MiTaPu (not a MARVEL run, no labels); D2H+ (no labels; manifest units wrong for 2
  sources); CO 12C-17O, 12C-18O, 13C-17O, 13C-18O (fewer than 10 negations, 1 unc column).

## Graph per run

- **Unedited graph:** every line, negated lines flipped back to positive, original uncertainty (column 2 for
  2-column files, the only column otherwise). Units come from the manifest, or are inferred from the published
  levels (N2O's 91 sources, CaOH's 1 PGOPHER tag); the inference rule is `infer_units`.
- **CaOH:** the copy works around a db_MARVEL parse bug (see README) and splits only the 206 rows where
  frequency and unc are glued.
- **Scope:** the largest connected component of the unedited graph (C2H4: the ground component, as in the
  blind test). Lines outside it are counted, not scored.

## Labels

- **X (excluded sources):** negated lines of a source whose every line is negated, plus C2H4's 58AlPl.
  Scored as their own set, secondary.
- **N (network negations, primary positives):** negated lines in the largest component that are not in X.
  - C2H4: the 544 key `negate` lines in the ground component, plus the 81 source negations outside 58AlPl.
- **Outlier negatives:** every other line in the largest component.
- **U (uncertainty raised, 2-column runs):** non-negated lines in the largest component with
  used > original·(1 + 1e-9).
  - Uncertainty negatives: non-negated lines with used = original.
  - Lowered lines (used < original; 636: 1309, others ≤ 35) are dropped from both sides.
  - C2H4: line-level `set_unc` (definition (a) of the blind test).
- **The audit's caveat on U, stated in advance:** in every 2-column run the published solve is fully
  consistent, and most raised lines sit at used ≈ |residual|. Examples: VO 1333/1337, CO 3768/3783,
  CO2 626 9483/11283, N2O 13429/15615.
  - So U is largely MARVEL 4's automatic uncertainty optimisation, not a curator decision.
  - C2H4's `set_unc` is the only curator uncertainty label in the set.
  - The replay baselines below test exactly this.

## Outlier scores (unedited graph, largest component)

| score | definition |
|---|---|
| (i) one-pass | \|r\|/unc from the Python-port weighted least-squares solve |
| (ii) studentized | \|r\| / (unc·sqrt(1 − h)), h from sparse `edge_leverages` |
| (iv) loop | K = \|N\| + \|X ∩ component\| times: drop the worst \|r\|/unc line, re-solve, keep the largest component. Score = −removal order. Never-removed lines and lines cut off by a removal tie at the bottom. Identical to the blind test's (iv). |
| (iv′) label-free loop, secondary | same loop, stopping when max \|r\|/unc ≤ 3 or after 3K removals |
| (R) rules | below; score = 1 if any applicable rule fires, else 0 |
| random | analytic expectation |

**Rules: generic, chosen by QN schema only, no per-molecule cases, no tuning.** Tokens a rule can't read
skip that rule for that line.

| rule | applies to (QN names present) | fires when |
|---|---|---|
| R1 ΔJ | every schema with J | \|J_u − J_l\| > 1 |
| R2 e/f parity | `ef` (CaOH, AlO, VO hfs, SO, PN, CO2, N2O) | ΔJ = 0 and e/f equal, or \|ΔJ\| = 1 and e/f differ |
| R2′ ± parity | `parity` (VO fine) | upper and lower parity equal |
| R3 linear l | `l2` (CO2) | J < l2 on either level, or l2 = 0 with `f` |
| R4 asym-top labels | `Ka`, `Kc` (HOCl, C2H4) | Ka + Kc ∉ {J, J+1}, or Ka > J, or Kc > J on either level |
| R5 asym-top dipole | `Ka`, `Kc` | ΔKa and ΔKc both even |

- C2H4's symmetry-block rule (QAB) is molecule-specific and is **not** included. It stays on the "missed"
  side by design.
- N2O's QN names are unverified (no manifest), so only R1 and R2 apply there. Its J is the 5th token, its
  `ef` the 4th, read off the file.

## Outlier metrics

- **Per run, on N ∪ negatives (X left out):**
  - AP (ties broken by line order);
  - recall@K with K = \|N\|;
  - recall in the top 1% of the pool;
  - precision of the loop's K removals against N ∪ X.
  - For (iv′): the number of removals, and its precision and recall.
- **Partition of N (headline).** Each positive goes in the first bucket that applies:
  1. **B, bridge:** leverage h ≥ 0.99 in the unedited component (original unc).
  2. **L, loop-caught:** removed within the loop's first K removals.
  3. **R, rule-caught:** flagged by an applicable rule.
  4. **H, headroom:** everything else.
  - **Reported twice, both primary:** L from (iv) (K from the labels) and L′ from (iv′) (label-free
    stop). Each gives pooled fractions and group CIs.
  - Also reported: B ∩ rule, L ∩ rule, and the same partition for X.
  - For C2H4, by key category: rule-shaped (QAB, QAI/QAM) vs outlier-shaped (QAC, QAJ, QAU), plus the 81
    source negations.
- **Pooling.** Each molecule group gets weight 1, split equally across its runs. Groups: CO2 isotopologues
  together, CO together, VO's 2 runs together, H3+ with H2D+.
  - Pooled value = weighted mean of per-run values.
  - 95% CI by cluster bootstrap over molecule groups (1000 resamples, seed 0).
  - Unweighted per-line pooled numbers are also reported, labelled as such.

## Uncertainty metrics (2-column runs; C2H4 via its key)

**Scores on the unedited graph, original unc:**
- `1/unc`;
- `|r|/unc` (i);
- studentized (ii);
- **per-source scatter** (the gscd-style data-driven estimate): s_src = 1.4826 · median(\|r\|/unc) over the
  source's lines in the component; every line of the source takes that value.

**Replay baselines (the no-model pipeline for uncertainty):** a MARVEL 4-style replay. Repeat {solve; every
line with \|r\| > unc gets unc = \|r\|} until no unc changes by more than 0.1%, or 100 rounds. The score
is the final unc / original.
- **P, pipeline:** replay on the loop (iv)'s pruned graph. Non-negated lines that the loop removed rank
  above all others.
- **P\*, oracle replay:** replay on the edited graph (curator negations removed). This measures how
  automatic U is once the negations are known; it is not a no-model baseline.

**Per run:**
- AUROC for U;
- Spearman ρ, among U positives, between the applied factor (used / original) and each score;
- precision and recall of P's raised set (factor > 1.001).

Pooled as for outliers (7 groups). The C2H4 line is the blind test's set_unc AUROC with the same scores, plus
P and P\*.

## Primary endpoints

1. **Partition of N, pooled with molecule weighting:** the fractions B, L, R, H with cluster-bootstrap CIs,
   once with L from (iv) and once with L′ from (iv′).
2. **Pooled outlier AP:** (iv) vs (i), each with its CI.
3. **Pooled uncertainty AUROC:** P vs the best of `1/unc`, (i), (ii) and per-source scatter. Plus the
   pooled Spearman ρ of P's factor with the applied factor.

Everything else is secondary. Anything not listed here is labelled post hoc.

## Interpretation, stated in advance

**Upper bound only.** H is an upper bound on what a learned model could add on top of the pipeline. A
headroom line may have been negated on external evidence that no structural model can see.

**Outlier recommendation for step 1** (real-label leave-one-molecule-out GNN, the loop as baseline, plus a
loop+GNN hybrid):
- **not worth running** if pooled H < 0.20;
- **worth running** if pooled H ≥ 0.40, and H is spread over at least 4 molecule groups with H ≥ 0.25;
- otherwise, a judgement call, stated with reasons.
- The thresholds apply separately to H from (iv) and H′ from (iv′). If the two verdicts disagree, the
  outcome is a judgement call, and the report says why.

**Uncertainty:**
- If P's pooled AUROC is ≥ 0.95 and ρ ≥ 0.8, U is algorithmic. A learned uncertainty head has no target in
  these runs beyond the replay. The only curator uncertainty label is C2H4's, a single molecule, too thin to
  train or validate step 1.
- If P falls well short, the shortfall is the uncertainty headroom.

**The loop's K uses the label count (as in the blind test).** That flatters the loop and shrinks H. The
(iv′) partition is the label-free counterpart.

## Threats to validity, stated in advance

- **1-column runs** (CN, CaOH, AlO, HOCl, CO 13C-16O) may already carry optimised uncertainties. CN and AlO
  re-solve with 0 inconsistent lines. If so, their "original" unc is inflated on exactly the lines MARVEL
  flagged, which hides some outliers from (i) and (iv).
- **Negation reasons are unknown outside C2H4.** Whole-source exclusion (X) is the only reason proxy. PN
  negates 52% of its lines (1382 in one whole source).
- **db_MARVEL version labels** are mostly `unknown`. The 2-column runs behave like MARVEL 4 (consistent
  final solve).
- **Inferred units** (N2O 1030 non-cm-1 lines) rest on the published levels. A wrong unit would show up as a
  failed re-solve; the audit's re-solve reproduces 100% of N2O levels.
- **Single-line removal loop:** at TiO / N2O scale it costs about 0.3–0.6 s per removal, so it runs as is (no
  batching).
