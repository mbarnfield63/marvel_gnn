# Step 2: how optimal are the cleaning methods? — pre-registration

**Written:** 2026-10-30, before any method is run for scoring. The only computation so far is a MILP timing
probe on 8 runs (k = 3, a 300 s limit, objective counts only); it touched no label and no score.
**Revised:** 2026-10-30, before any scoring run, with vaultagent-26's two additions (re-admission "+R"
variants with a swamping secondary; removal attribution) and three implementation details the first
version left open (marked *detail*). Previous version sha256 daa141b6…057c.
**Task:** vaultagent-26, approved by the user.
**Question (the user's):** does greedy "remove the worst line, re-solve" reach a real optimum — the most
lines kept and levels validated, with accurate level uncertainties? Or does a better method exist?
- Every method, the curators included, is scored against a stated objective with label-free accuracy
  measures. Agreement with curators is secondary.
- Known greedy failure modes: swamping (a good line removed because a bad neighbour pulled it off) and
  masking (a bad line hidden by another).
- Removing a line vs inflating its uncertainty is a real trade-off and is scored as one.

## Data (unchanged from steps 0 and 1′)

- **Runs:** the 21 outlier-usable runs in 13 molecule groups (`dbmarvel_headroom.records()`):
  - CN, CaOH, AlO, VO ×2, HOCl, H3+, SO, TiO, PN, N2O, C2H4;
  - CO2 ×6; CO ×2.
- **Starting graph:** the unedited graph (every line, negations flipped back in, original unc, manifest or
  inferred units), restricted to its largest component.
- **Notation:** n_lines(run) and n_levels(run) count that component. "Original unc" is u⁰.

## Common definitions

**Solve.** Weighted least squares, w = 1/u². The lowest-energy level is pinned (the reduced Laplacian; same
energies as `core.solver`). z_i = |ν_i − (E_u − E_l)| / u_i.

**Consistency threshold k.**
- Primary k = 3; sensitivity runs at k = 2 and k = 5.
- Every removal method ends k-consistent: max z ≤ k on its final network. "Finish" means: after its own
  removals, the method runs M1 on what is left until max z ≤ k.

**Final network of a method:**
- the largest connected component of its kept lines, with its unc vector u (u⁰, or inflated);
- lines cut off from that component count as removed;
- removed-line count = n_lines − kept lines.

**Validated level:** a level incident to at least one non-bridge edge of the final network's simple graph
(parallel lines collapsed), i.e. on a cycle. Orphaned levels = levels of the unedited component that are not
in the final network.

**Inflated line:** u > u⁰·(1 + 1e-9). Reported as a count and as the median factor u/u⁰.

## Methods

| id | method | definition |
|---|---|---|
| M0 | curators | the published edited network: negated lines (N and X) removed, used unc (the deliverable's optimised unc for C2H4; the single column for 1-column runs). Largest component of what is left. A contender, not truth; not forced k-consistent (its max z is reported). |
| M1 | greedy \|r\|/u | remove the max-z line and re-solve the largest component, until max z ≤ k. No cap. |
| M2 | greedy studentized (Baarda data snooping) | remove the max z/sqrt(1 − h) line (h from sparse `edge_leverages`, unc floored at 1e-6 cm-1 for h only, as in step 0), re-solve, until max z ≤ k. Lines with h ≥ 0.999 are never removed by this rule: they can't be tested. The final check is still max z ≤ k, so a finish M1 step runs if bridges still violate it. |
| M3a-H, M3a-T | robust IRLS, remove | IRLS with weights w_i·ψ(z_i)/z_i. Huber, c = k: weight = min(1, k/z). Tukey, c = 2k: weight = (1 − (z/2k)²)² for z < 2k, else 0. Up to 100 iterations, stop when max \|ΔE\| < 1e-3·min(u). Then remove lines with robust z > k, LS re-solve, finish. |
| M3b-H, M3b-T | robust IRLS, inflate | same IRLS; keep every line with u = u⁰/√weight. Tukey lines with weight 0 are removed (infinite unc). No finish step. |
| M4 | weighted L1 | minimise Σ\|r_i\|/u_i over the energies (`linprog`, HiGHS). Remove lines with z_L1 > k, LS re-solve, finish. |
| M5 | robust + per-source variance components | alternate Huber IRLS (c = k) with Helmert-style per-source scale updates. Then remove robust z > k (on the scaled u), LS re-solve with the scaled u, finish (M1 on the scaled u). Kept lines carry u = s_src·u⁰. Details below. |
| M6-1, M6-2 | remove then replay | MARVEL 4-style replay (raise u to \|r\| wherever \|r\| > u, re-solve, until no u moves by > 0.1%, 100 rounds max) on M1's and M2's final networks. |
| M6-0 | inflate only | the same replay on the full unedited component: no removals. This is the extreme of the remove-vs-inflate axis. |

**M5 details.** Start with s_src = 1. Repeat until max \|Δs\|/s < 1% (20 rounds max):
1. Huber IRLS with u = s_src·u⁰.
2. For each source with redundancy Σ(1 − h_i) ≥ 5:
   s_src ← s_src · sqrt( Σ w_i z_i² / Σ w_i (1 − h_i) ),
   using that source's lines, robust weights w and leverages h at the current solution. Bounded to
   [0.5, 100].

Sources with redundancy below 5 keep s = 1.

**+R variants (re-admission): M1+R, M2+R, M3a-H+R, M3a-T+R, M4+R, M5+R.**
- Run after the base method finishes.
- Candidates: the method's removed lines (explicit or cut off) whose two levels are both in the current
  network.
- Each candidate's predicted z: \|ν − (E_u − E_l)\| / u, using the line's own u under that method (u⁰, or the
  scaled u for M5).
- Loop:
  1. Take the candidate with the smallest predicted z, among those with predicted z ≤ k.
  2. Add it and re-solve.
  3. If max z ≤ k still holds on the whole network, keep it. Otherwise drop it for good.
  4. Recompute the predicted z of the remaining candidates and repeat.
  5. Stop when no untried candidate has predicted z ≤ k.
- +R variants enter the Pareto front and the winner selection like any other method.

**Removal attribution (every removal method):** each removed line is counted under exactly one of:
- (i) removed by the method's own rule;
- (ii) removed by the M1 finish step;
- (iii) cut off from the largest component;
- (iv) for +R variants, the number re-admitted.

**Implementation details:**
- *Detail:* all uncertainties are floored at UNC_FLOOR = 1e-6 cm-1 for every method and metric (the parser's
  floor; step 0 needed it for leverage). The rows below that floor: CO2 626 1068, N2O 384, SO 316, and fewer
  elsewhere.
- *Detail:* Tukey IRLS starts from the converged Huber solution, the standard way to avoid a non-robust start.
- *Detail:* in IRLS, lines with weight 0, or outside the largest positive-weight component, are treated as
  removed.

**EXACT (MILP, `scipy.optimize.milp`, HiGHS):**
- Objective: minimise Σ b_i, with b_i binary, subject to \|ν_i − (E_u − E_l)\| ≤ k·u⁰_i + M_i·b_i.
- E is continuous: the ground level is fixed at 0, and the others bounded to [−S, 2S], with S = max LS energy
  + 1 cm-1. Big-M: M_i = ν_i + 2S.
- Runs at k = 3 on every run, with a 2 h time limit per run.
- Reported: incumbent, dual bound, and status (optimal or gap). The MILP doesn't enforce connectivity: a
  cut-off level's lines are free. So it is compared with each method's *explicit* removals (lines the method
  chose to remove, not counting cut-off ones), on the same u⁰.
- **Probe timings** (300 s limit; label-free; to choose the limit):
  - solved to optimality: H2D+ 202 lines 0.0 s; CO 13C-16O 3,397 lines 0.4 s; CaOH 3,240 lines 1.4 s;
    H3+ 1,640 lines 30 s; CO 12C-16O 19,233 lines 97 s; VO hfs 7,921 lines 173 s; VO 9,164 lines 268 s;
  - **not** solved: PN, 5,920 lines, hit the limit with incumbent 347 and dual bound 129.
  - Size alone doesn't predict hardness; contamination does.
  - **Size rule, stated in advance:** every run gets the MILP with a 2 h limit. Runs without proven
    optimality report the gap to the dual bound, which still bounds each method's excess removals from below.

## Held-out accuracy (the core measure)

**Folds.** One fixed fold id per line of the unedited component: fold = hash(line ref, seed 0) mod 10. It is
the same for every method, so the comparisons are paired.

**For each method and fold f:**
- Train on the final network minus fold f, with the method's u. No re-cleaning.
- Take the largest component of that training graph and solve.
- Score each held-out line of the method's final network that is in fold f and has both levels in the
  training component:
  - pred = E_u − E_l;
  - σ²_pred = e^T G e, with e = e_u − e_l and G the inverse of the reduced Laplacian (computed exactly by
    batched sparse solves on the training factor);
  - s² = u_i² + σ²_pred, where u_i is the method's unc for that line;
  - z = (ν − pred)/s;
  - NLL_i = ½ log(2π s²) + ½ z².
- Held-out lines whose levels fall outside the training component can't be scored. They are counted
  ("disconnected"), not scored.

**Per method and run:**
- **NLL:** mean over scored lines (primary accuracy metric).
- **Coverage:** \|z\| ≤ 1 and \|z\| ≤ 2 (nominal 0.683 and 0.954).
- **Per-source calibration:** mean z² per source (sources with ≥ 10 scored lines); reported as the share of
  sources inside [0.5, 2].
- **Disconnected:** the count.

**Common-set NLL (secondary, the fair comparison).** The same quantities over C = the lines kept by every
method at that k, M0 included. Each method predicts them from its own network, using its own u for that line.
This removes the advantage a method gets from scoring only the lines it kept.

**Sharpness:** the median formal level σ, sqrt(G_jj), on each method's full final network (ground pinned).
Reported alongside NLL, which already rewards calibration and sharpness together.

## Primary comparison (k = 3)

**Per run, against M1:**
- ΔNLL = NLL(method) − NLL(M1), each on its own final network;
- Δkept = kept-line fraction difference;
- Δvalidated = validated-level fraction difference (validated levels / n_levels).

**Pooling:** molecule-weighted (each group weight 1, split over its runs). 95% cluster bootstrap over the 13
groups (1000 resamples, seed 0). The same comparison is also made against M0.

**Method X dominates method Y** when all of these hold:
- **NLL no worse:** the upper CI bound of pooled ΔNLL(X − Y) < +0.05 nats;
- **lines and levels no lower:** pooled Δkept ≥ −0.002 and Δvalidated ≥ −0.002 (point estimates; a 0.2%
  tolerance for ties);
- **strictly better on at least one axis:** the ΔNLL CI is entirely below 0, or Δkept or Δvalidated has its CI
  lower bound > 0.

**Pareto front:** the methods not dominated by any other method at k = 3.

**Winner:**
- If exactly one front method also dominates M1, it is recommended.
- If several do, the tie-breaks apply in this order: lowest common-set NLL, then fewest inflated lines, then
  simplest.
- If none dominates M1, greedy M1 is declared adequate.

## Secondary

0. **Swamping (all removal methods).** After the method finishes, predict each removed line whose levels are
   both in the final network. Report how many have predicted \|z\| ≤ k: likely swamping victims. Reported
   per method as a count and as a share of explicit removals.

1. **Gap to the exact minimum (k = 3):** each removal method's explicit removals (attribution (i) + (ii), net
   of re-admissions for +R) minus the MILP incumbent,
   pooled as a share of n_lines, plus each run's gap to the dual bound. **Greedy is called near-optimal** if
   M1's pooled excess over the MILP optimum is ≤ 1% of lines and ≤ 25% of the MILP's own removal count.
2. **k sensitivity:** the primary table at k = 2 and k = 5.
3. **Agreement with the curators (continuity with steps 0 and 1′):**
   - for removal methods, precision and recall of the removed set against N ∪ X;
   - for inflation methods, AP_ties of the inflation factor against N ∪ X.
4. **Runtime** per method and run.
5. **Diagnostics:**
   - max z on M0's network;
   - Tukey vs Huber;
   - M5's estimated source scales, and their correlation with C2H4's key `scale` rows. That is post hoc
     context only; C2H4's 5 scaled sources are too few for a test.

## Output, stated in advance

- **Headline:** is greedy M1 near-optimal (secondary 1)? Which methods are on the Pareto front?
- **Recommendation for marvel-network on C2H4 v2:** a method, a k, and a remove-vs-inflate policy, chosen
  from the front by the rule above. Advisory only, as always.

## Computation

- **Code:** numpy/scipy only. One script, `scripts/cleaning_optimality.py`, with a per-(run, method, k)
  cache.
- **Where it runs:** greedy loops and the MILP may run on Hypatia (CPU), one job per run × method × k, with a
  24 h wall cap. Anything unfinished is reported as incomplete; nothing is extrapolated.
- **Speed:** loops use a vectorised solver (array inputs, one `splu` per step) that gives the same energies as
  `core.solver`. Its agreement with `solve_energies` is checked on every run before use (max \|ΔE\| < 1e-8
  cm-1).

## Threats to validity, stated in advance

- **Own-set NLL favours aggressive removal**: a method is scored only on the lines it kept. Kept fraction
  sits in the Pareto comparison, and the common-set NLL is the fair accuracy check.
- **Held-out lines carry the method's own u.** Inflation shows up as both less sharp and better calibrated;
  NLL arbitrates.
- **1-column runs may already carry optimised unc** (CN, AlO, HOCl, CaOH, CO 13C-16O). Their u⁰ is then
  already inflated where MARVEL flagged lines, which favours "do nothing".
- **The MILP ignores connectivity and uses k·u⁰.** It bounds the removal count only, not accuracy.
- **13 groups:** wide CIs. CO2 is one group of 6 runs.
