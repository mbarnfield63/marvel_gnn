# Step 2: how optimal are the cleaning methods?

**Task:** vaultagent-26, approved by the user. **Status:** done; pre-registered winner overturned post hoc (scoring leak, see 3′).
**Pre-registration:** `PREREGISTRATION.md`, sha256
`3acb8d08af323da1e7a7ec3188e8f80d56a3be213b8919389014b5eced85cc43` (the revised version with the +R
variants, removal attribution and three implementation details; written before any scoring run).
**Files:**
- scripts: `scripts/cleaning_optimality.py` (stages), `scripts/cleaning_report.py` (aggregation),
  `scripts/cleaning_posthoc.py` (post hoc MILP+F check, not pre-registered), `job_cleaning.sh` (Hypatia array);
- outputs: `results.json`, `results.txt` (pre-registered); `posthoc_milp_finish.json`, `posthoc_nested.json`,
  `posthoc_nested.txt` (post hoc);
- caches: `data/dbmarvel/cleaning/` (`run_*`, `out_*`, `score_*`, `milp_*`; post hoc `milpx_*`, `milpe_*`, `nested_*`).
**Compute:** Hypatia array 1398081 (534 tasks, CPU-only on the GPU partition, snapshot
`runs/20261030-131957`), plus test task 1398080 and local runs. Longest task 2 h 30 min.

## Deviations from the pre-registration

1. **Run count.** There are **20** outlier-usable runs in 13 groups, not 21. The pre-registration (and the
   step 0 and step 1′ reports) said 21, but its own list sums to 20 (12 single runs, CO2 ×6, CO ×2). Every
   number here is over the 20 runs.
2. **Solver check.** The pre-registered check (max |ΔE| < 1e-8 cm-1 against `core.solver.solve_energies`)
   is unattainable at float precision on several runs (CN 9.9e-8 cm-1; CaOH 0.14σ; TiO 18σ; CO2 626 54σ).
   The vectorised solver gained one iterative-refinement step (weights span up to 1e14), and a run passes if
   any of these holds: |ΔE| ≤ 1e-7 cm-1, |ΔE| ≤ 0.01σ, or its χ² ≤ core's χ²·(1 + 1e-7). All 20 pass.
   - **Finding:** `core.solver` itself misses the LS minimum on TiO (χ² 1,109,487 vs 1,109,146 here) and
     slightly on CaOH. **Effect on step 0:** TiO's one-pass |r|/u AP is 0.4106 with the exact solve vs 0.4115
     in step 0. Step 0's conclusions stand.
3. **MILP bound rounding.** The first 5 MILP files (probe-era) rounded HiGHS's dual bound with
   ceil(raw − 1e-6), which gives incumbent + 1 on 4 optimal runs (e.g. 9.0000035 → 10). The script now uses
   ceil(raw − 1e-3), and the report caps the bound at the incumbent. Nothing else uses the bound.
4. **MILP failure on 16O-12C-18O.** HiGHS returns "Solve error" (status 4) with no incumbent and no bound,
   on Hypatia after 25 min and again locally after 4 min. Reported as **incomplete**; 19 of 20 runs have a
   MILP result.
5. **Where it ran.** Everything ran on Hypatia CPU nodes (allowed by the pre-registration), except the 200
   already-cached local results and one rerun task (CN M0 chain).

Implementation details fixed in the revised pre-registration (UNC_FLOOR on all u; Tukey started from the
Huber solution; zero-weight lines treated as removed) were applied as written.

## Headline (pre-registered)

**1. Is greedy near-optimal? The pre-registered test says no, but the test is invalid; the post hoc answer
is "mostly yes, with clear exceptions".**
- *As computed:* on the 12 runs the MILP "solved to optimality", M1's explicit removals exceed the MILP count
  by 2.1% of lines [1.0, 3.3] (bar ≤ 1%) and by 17× the MILP's count (bar ≤ 0.25×). Verdict: **False**.
- *But the MILP optima are not feasible* (post hoc check): on all 11 runs checked, the MILP's own energies put
  8–1,164 of its "kept" lines above k·u⁰, by up to 0.125 cm⁻¹. HiGHS accepts a binary of about 1e-6 as 0, and
  with big-M ≈ 1e5 cm⁻¹ that opens about 0.1 cm⁻¹ of slack, against u⁰ down to 1e-6 cm⁻¹. The MILP counts
  are therefore lower bounds on the minimum, not the minimum, and the pre-registered gap overstates greedy's
  excess.
- *Post hoc, against the best k-consistent LS set found* (the better of M1 and "MILP+F" = MILP set + M1
  finish), on 11 runs in 7 groups:
  - M1 is the best set found on 7 runs: it ties MILP+F on 5 CO/CO2 runs and beats it by 1 line on CaOH and
    by 3 on H3+;
  - MILP+F beats M1 on 4 runs: HOCl (52 vs 54), AlO (617 vs 627), VO (372 vs 419) and VO hfs (42 vs 111).
  - Pooled, M1's excess over the best set found is 0.11% of lines and 13% of the best removal count. Both
    are inside the pre-registered bars (≤ 1%, ≤ 25%).
  - The true optimum may lie below both sets, so this is a lower bound on greedy's gap, not a proof of
    near-optimality. VO hfs alone shows greedy can remove 2.6× more than needed.

**2. Pareto front at k = 3:** M0 (curators), M6-0 (inflate only), M3b-T (Tukey inflate), M6-1 (M1 + MARVEL 4
replay), M5+R. Two front methods dominate M1: **M6-1** and **M3b-T**.

**3. Winner (pre-registered tie-break: lowest common-set NLL): M6-1.** Greedy M1 to k = 3, then the MARVEL 4
replay (raise u to |r| where |r| > u, re-solve, repeat).

**3′. Post hoc correction: the win is a scoring leak.** The pre-registered score gives each held-out line
the method's final u. For inflating methods that u was set from the line's own full-network residual
(replay: u_i ≥ |r_i|), so held-out z is pulled toward 1 by construction. A nested check (VaultAgent's review)
redoes the inflation per fold on the training lines only:
- M6-1's ΔNLL vs M1 goes from −0.137 to +0.007 [−0.001, +0.021] (held-out u = u⁰) or +0.001 [−0.003, +0.007]
  (u⁰ × the source's median training factor).
- M3b-T goes from −0.026 to +0.142 [+0.064, +0.225]: worse than M1.
- **On leak-free numbers nothing dominates M1. Greedy M1 at k = 3 is adequate** (the pre-registered rule's
  outcome when no front method dominates M1). See "Post hoc: nested held-out check".

## Primary table (k = 3)

Pooled over 13 groups, molecule-weighted. NLL in nats per scored held-out line (lower is better).

| method | NLL | common-set NLL | kept | validated | inflated lines | cov \|z\|≤1 | cov ≤2 |
|---|---|---|---|---|---|---|---|
| M0 curators | −3.786 | −3.878 | 0.9458 | 0.7565 | 2905 | 0.895 | 0.989 |
| M6-0 inflate only | −3.159 | −3.278 | 1.0000 | 0.7854 | 7270 | 0.912 | 1.000 |
| M1 greedy \|r\|/u | −3.722 | −3.817 | 0.9671 | 0.7749 | 0 | 0.823 | 0.953 |
| M2 studentized | −3.783 | −3.838 | 0.9632 | 0.7717 | 0 | 0.836 | 0.964 |
| M3a-H Huber remove | −3.736 | −3.820 | 0.9582 | 0.7698 | 0 | 0.827 | 0.956 |
| M3a-T Tukey remove | −3.752 | −3.828 | 0.9627 | 0.7728 | 0 | 0.829 | 0.958 |
| M3b-H Huber inflate | +85.6 | −3.683 | 1.0000 | 0.7854 | 933 | 0.775 | 0.902 |
| **M3b-T Tukey inflate** | −3.748 | −3.848 | 0.9781 | 0.7747 | 25942 | 0.827 | 0.960 |
| M4 weighted L1 | −3.805 | −3.842 | 0.9556 | 0.7638 | 0 | 0.850 | 0.972 |
| M5 Huber + Helmert | −3.175 | −3.256 | 0.9910 | 0.7820 | 12582 | 0.873 | 0.981 |
| **M6-1 M1 + replay** | −3.860 | **−3.899** | 0.9671 | 0.7749 | 2648 | 0.888 | 0.999 |
| M6-2 M2 + replay | −3.872 | −3.903 | 0.9632 | 0.7717 | 2359 | 0.892 | 1.000 |
| +R variants | ≈ base | ≈ base | +0.0000 to +0.0018 | ≈ base | | | |

Against M1 (ΔX − M1, 95% cluster-bootstrap CI):

| method | ΔNLL | Δcommon NLL | Δkept | Δvalidated |
|---|---|---|---|---|
| M6-1 | −0.137 [−0.218, −0.065] | −0.083 [−0.130, −0.039] | 0 | 0 |
| M3b-T | −0.026 [−0.049, −0.004] | −0.031 [−0.048, −0.014] | +0.0110 [+0.0040, +0.0184] | −0.0001 [−0.0015, +0.0013] |
| M6-2 | −0.150 [−0.234, −0.072] | −0.086 [−0.134, −0.041] | −0.0038 | −0.0031 |
| M2 | −0.061 [−0.095, −0.028] | −0.021 | −0.0038 | −0.0031 |
| M4 | −0.083 [−0.132, −0.039] | −0.025 | −0.0115 | −0.0111 |
| M0 | −0.064 [−0.243, +0.155] | −0.061 [−0.138, +0.011] | −0.0212 [−0.086, +0.022] | −0.0184 |
| M5 | +0.548 [+0.052, +0.999] | +0.561 | +0.0239 | +0.0072 |
| M6-0 | +0.563 [+0.172, +1.012] | +0.539 | +0.0329 | +0.0105 |

**Reading:**
- **Inflation "wins" here are leaked (see 3′).** M6-1 keeps exactly M1's lines; its 0.14 nats come from
  held-out lines carrying u inflated from their own residuals. M1 with u⁰ already covers above nominal at 1σ
  (0.82 vs 0.68) and at nominal at 2σ (0.95); the replay pushes this to 0.89 / 1.00.
- **Removing more lines buys NLL, not optimality.** M2, M4 and M6-2 beat M1 on NLL only by removing more;
  they fail the "no fewer lines" leg and are not on the front.
- **Inflating everything (M6-0) or scaling whole sources (M5) costs 0.55 nats.** Most M5 scales hit the 100
  cap: per-source Helmert scales, driven by a few bad lines, blur the good ones.
- **M3b-H (Huber inflate) fails badly:** u grows only as √z, so gross outliers keep huge z (own-set NLL +86).
- **M3b-T keeps 1.1% more lines than M1 at slightly better NLL.** It is the only method that keeps more lines
  than M1 without losing accuracy (M5 and M6-0 keep more but lose 0.55 nats).
- **Curators (M0) are not dominated and dominate nothing.** Their NLL CI vs M1 spans 0 widely; they keep
  2.1% fewer lines (AlO, PN and TiO have large curator negations).

**Dominance (k = 3).** M6-1 dominates M1, M2, M3a-*, M4, M6-2 and their +R variants. M3b-T dominates M1, M2,
M3a-* and their +R. Full list in `results.txt`.

## Secondary results

**0. Swamping and re-admission.**
- Removed lines that fit the final network within k (likely swamping victims): M1 0 (pooled), M2 2, M3a-H 28,
  M4 34, M5 4. M0 177: many curator negations fit the curated network.
- The +R variants re-admit almost nothing: M1+R 0 lines pooled, M2+R 2, M3a-H+R 27, M4+R 35, M5+R 4. Their
  effect on every metric is within ±0.002.
- So greedy's excess removals (headline 1) are **not** lines that fit back in afterwards.

**1. MILP gap (k = 3).**

| run | lines | MILP incumbent / bound | status | M1 explicit | M5 explicit | M0 |
|---|---|---|---|---|---|---|
| CN | 40,243 | 78 / 59 | time limit | 111 | 85 | 183 |
| CaOH | 3,240 | 9 / 9 | optimal | 15 | 7 | 16 |
| AlO | 23,934 | 17 / 17 | optimal | 627 | 413 | 1499 |
| VO | 9,164 | 55 / 55 | optimal | 419 | 81 | 54 |
| VO hfs | 7,921 | 10 / 10 | optimal | 111 | 57 | 41 |
| HOCl | 21,462 | 52 / 52 | optimal | 54 | 98 | 147 |
| H3+ | 1,640 | 44 / 44 | optimal | 70 | 35 | 60 |
| SO | 49,206 | 144 / 60 | time limit | 149 | 18 | 388 |
| CO2 636 | 20,670 | 78 / 78 | optimal | 1239 | 118 | 29 |
| TiO | 62,801 | 411 / 155 | time limit | 533 | 540 | 1416 |
| PN | 5,920 | 352 / 160 | time limit | 664 | 68 | 3077 |
| CO2 626 | 44,333 | 210 / 124 | time limit | 3197 | 649 | 349 |
| CO2 638 (16O-13C-18O) | 12,009 | 18 / 18 | optimal | 398 | 80 | 21 |
| CO2 727 (12C-17O2) | 10,140 | 40 / 40 | optimal | 744 | 184 | 85 |
| CO 12C-16O | 19,233 | 7 / 7 | optimal | 794 | 136 | 12 |
| CO2 627 (16O-12C-17O) | 16,076 | 44 / 44 | optimal | 857 | 213 | 103 |
| CO 13C-16O | 3,397 | 15 / 15 | optimal | 32 | 18 | 15 |
| N2O | 67,374 | 267 / 160 | time limit | 3495 | 773 | 634 |
| C2H4 | 50,294 | 245 / 215 | time limit | 3180 | 513 | 679 |
| CO2 628 (16O-12C-18O) | 33,473 | — | HiGHS solve error | | | |

The MILP columns are as computed. The post hoc section shows that these counts are not feasible removal
sets.

**2. k sensitivity.**
- k = 2: front {M0, M6-0, M3b-T, M6-1, M5+R}; only M6-1 dominates M1. M3b-T's pooled NLL is invalid at k = 2
  (see below).
- k = 5: front {M0, M6-0, M3b-T, M5+R}; only M3b-T dominates M1. M6-1 and M6-2 dominate each other (the
  rule's 0.2% tie tolerance on lines allows that), so neither is on the front. Both still dominate M1.
- The winner by the pre-registered rule is the same at k = 3 and k = 2 (M6-1); at k = 5 it is M3b-T, the only
  front method left that dominates M1. Both are inflating methods, so the scoring leak (3′) applies; the
  nested check was run at k = 3 only.
- **M3b-T at k = 2 breaks numerically on 16O-12C-18O:** Tukey weights reach ~1e-19 (u inflated 3.6e9×,
  weight span 1.3e25), past float64 LS. Held-out z reaches 5e5 on 2,439 lines; that one run sets the pooled
  NLL to 1.3e5. At k = 3 the span there is 2.4e11 and the run is fine. Inflate-by-Tukey needs a weight floor
  (below it, remove) before it can be deployed.

**3. Agreement with the curators (k = 3, pooled).**
- Removal methods, precision / recall against N ∪ X: M1 0.61 / 0.59; M2 0.57 / 0.55; M3a-T 0.57 / 0.63;
  M4 0.56 / 0.61; M5 0.66 / 0.38.
- Inflation AP (tie-aware, inflation factor vs N ∪ X): M3b-T 0.565; M3b-H 0.560; M6-1 0.524; M6-0 0.437.

**4. Runtime (pooled mean per run).** M2 960 s (leverage every step); M4 170 s; M1 115 s; M5 93 s; IRLS
methods 7–14 s; replay 3–4 s; +R ≤ 8 s. MILP: 4 s to > 2 h (7 runs hit the limit).

**5. Diagnostics.**
- **M0 max z:** ≈ 1 on 15 runs (MARVEL 4's replay); HOCl 2.9, PN 1.4, CO 13C-16O 6.2, TiO 14.0, C2H4 14.4.
- **Tukey vs Huber.** Remove: Tukey has the better NLL (ΔNLL vs M1 −0.030 vs −0.014), keeps more lines
  (0.9627 vs 0.9582), orphans fewer levels (28 vs 61 pooled) and dominates Huber. Inflate: Tukey is usable,
  Huber is not (primary table).
- **M5 C2H4 source scales vs the key's `scale` rows** (post hoc context, 5 sources): 02MoKoBuHe 10.1 (key 3),
  83DaPiFaDe 100 (key 5), 19UlBeGrRa 2.0 (key 2), 24BeVaTuBo 100 (key 2), 25BeBeReCa 100 (key 2). Direction
  right on all 5; magnitude right only where the key is small. The 100 cap binds on 3 of 5.

## Post hoc: how much of the MILP gap is real? (not pre-registered)

Script: `scripts/cleaning_posthoc.py`; output `posthoc_milp_finish.json`. Run after the pre-registered results
were in, because the MILP gap looked implausible: 794 removals for M1 against 7 for the MILP on CO 12C-16O.
Covers the 11 runs with a reproducible optimal MILP; 16O-13C-18O was optimal on Hypatia but HiGHS gave
"Solve error" on both local reruns.

**A. Are the MILP solutions feasible?** The MILP was rerun with its energies saved. Each "kept" line was then
checked against |ν − (E_u − E_l)| ≤ k·u⁰.

**B. What does the MILP's set need to satisfy greedy's own criterion?** LS-solve the MILP's kept set, then
run the M1 finish until the LS max z ≤ k ("MILP+F"). MILP+F is a valid k-consistent LS network; it is scored
like any method.

| run | lines | MILP removed | (A) kept lines violated | worst violation cm⁻¹ | (B) LS violations on MILP set | MILP+F removals | M1 removals | NLL MILP+F / M1 | validated MILP+F / M1 |
|---|---|---|---|---|---|---|---|---|---|
| CaOH | 3,240 | 9 | 8 | 0.068 | 7 | 16 | **15** | −3.133 / −3.133 | 0.5795 / 0.5801 |
| AlO | 23,934 | 17 | 361 | 0.117 | 726 | **617** | 627 | −2.728 / −2.728 | 0.8690 / 0.8688 |
| VO | 9,164 | 55 | 299 | 0.077 | 580 | **372** | 419 | −2.169 / −2.181 | 0.7478 / 0.7436 |
| VO hfs | 7,921 | 10 | 44 | 0.024 | 77 | **42** | 111 | −4.992 / −4.994 | 0.5883 / 0.5829 |
| HOCl | 21,462 | 52 | 34 | 0.014 | 0 | **52** | 54 | −5.660 / −5.661 | 0.8386 / 0.8383 |
| H3+ | 1,640 | 44 | 49 | 0.032 | 42 | 73 | **70** | −3.404 / −3.396 | 0.4496 / 0.4525 |
| CO2 636 | 20,670 | 78 | 1,164 | 0.043 | 1,515 | 1,239 | 1,239 | −5.150 / −5.150 | 0.7002 / 0.7002 |
| CO2 727 | 10,140 | 40 | 830 | 0.014 | 983 | 744 | 744 | −6.041 / −6.041 | 0.6957 / 0.6957 |
| CO 12C-16O | 19,233 | 7 | 904 | 0.125 | 844 | 794 | 794 | −5.574 / −5.574 | 0.8998 / 0.8998 |
| CO2 627 | 16,076 | 44 | 841 | 0.027 | 1,188 | 857 | 857 | −5.829 / −5.830 | 0.6964 / 0.6964 |
| CO 13C-16O | 3,397 | 15 | 21 | 0.003 | 18 | 32 | 32 | −5.946 / −5.946 | 0.8876 / 0.8876 |

**Findings:**
- **(A) The pre-registered MILP is unsound at this precision.** Every checked "optimum" breaks its own
  constraints, by up to 0.125 cm⁻¹, because of big-M times HiGHS's integrality tolerance. Its counts are lower
  bounds on the true minimum (the slack only makes the problem easier), not the minimum. MILP removals plus
  violated lines is a feasible set; on the existence criterion it beats M1 on AlO (378), VO (354) and VO hfs
  (54) and loses elsewhere.
  - *Fix, untested:* a solver with indicator constraints, or a much tighter per-line M_i (bound the energies
    to a few cm⁻¹ around the LS solution). scipy's `milp` exposes neither an integrality tolerance nor
    indicator constraints.
- **(B) The two criteria differ.** Even a feasible MILP set is not k-consistent under LS. LS spreads residuals
  differently from the min-max energies the MILP finds. Only HOCl's MILP set passes as is. M1's criterion is
  the one MARVEL uses, so (B) is the relevant comparison.
- **Greedy against the best LS-consistent set found:** see headline 1. Greedy is optimal-as-far-as-found on 7
  of 11 runs. It is beaten on VO and VO hfs, where a different removal set keeps 47 and 69 more lines, with
  the same NLL (within 0.012) and more validated levels.
- **On the five CO/CO2 runs MILP+F ends on M1's exact set.** The M1 finish from the MILP set converges to the
  same network. Greedy there is at least locally stable.
- **NLL is insensitive to which set is removed** (within 0.012 on every run). The choice between near-equal
  removal sets is about lines kept, not accuracy.

## Post hoc: nested held-out check (not pre-registered; VaultAgent's review)

**Why.** `cleaning_optimality.py` scores held-out line i with s² = u_i² + σ²_pred, where u is the method's
final u. For M6-x, M3b-T and M0 that u was fitted on the full network, including line i's own residual. Only
the change in training weights is a fair gain; the held-out line's own inflation is a leak.

**How** (`scripts/cleaning_posthoc.py nested`, output `posthoc_nested.{json,txt}`):
- Each method's removal set is fixed as computed.
- Per fold, the inflation is redone on the training lines only, starting from u⁰:
  - M6-1 and M6-2: replay;
  - M3b-T: Huber then Tukey IRLS;
  - M0: replay of the curated set from u⁰. This is a stand-in, since MARVEL 4's own optimisation can't be
    rerun.
- Held-out lines are then scored with (i) u⁰, or (ii) u⁰ × the median u/u⁰ of training lines from the same
  source (1 if none).
- Version (i) isolates the training-weight effect alone. Check: M1 reproduces its pre-registered NLL
  exactly.

**k = 3, pooled over 13 groups** (common set = lines scored for all five methods):

| method | (i) NLL | (i) common | cov1 / cov2 | (i) ΔNLL vs M1 | (ii) ΔNLL vs M1 | (ii) Δcommon vs M1 | C2H4 (ii) NLL / common |
|---|---|---|---|---|---|---|---|
| M1 | −3.722 | −3.794 | 0.823 / 0.953 | 0 | 0 | 0 | −6.493 / −6.560 |
| M6-1 | −3.715 | −3.791 | 0.827 / 0.953 | +0.007 [−0.001, +0.021] | +0.001 [−0.003, +0.007] | +0.001 [−0.004, +0.009] | −6.495 / −6.563 |
| M6-2 | −3.773 | −3.810 | 0.838 / 0.963 | −0.051 [−0.081, −0.024] | −0.056 [−0.086, −0.026] | −0.018 [−0.032, −0.007] | −6.568 / −6.582 |
| M3b-T | −3.581 | −3.770 | 0.811 / 0.936 | +0.142 [+0.064, +0.225] | +0.128 [+0.060, +0.203] | +0.022 [+0.005, +0.046] | −6.311 / −6.535 |
| M0 (stand-in) | −2.704 | −3.724 | 0.841 / 0.950 | +1.018 [+0.252, +1.838] | +0.846 [+0.157, +1.616] | +0.062 [−0.048, +0.190] | −5.071 / −6.436 |

Coverage shown for version (i); version (ii) is within 0.005.

**Dominance on these numbers** (lines and levels as pre-registered): M1, M6-1, M6-2 and M3b-T each dominate
the M0 stand-in. **Nothing dominates M1.**
- **The M0 stand-in is a weak proxy.** It replays the curated set from u⁰; it is not MARVEL 4's real
  uncertainty optimisation, which can't be rerun per fold. "M1 dominates M0" holds only against this
  stand-in.

**Reading:**
- **Answer to "accurate level uncertainties": the original uncertainties are already conservative after
  greedy removal.** M1 with u⁰ covers 0.82 of held-out lines within 1σ (nominal 0.68) and 0.95 within 2σ
  (nominal 0.95). Inflation adds nothing on held-out lines.
- **The replay adds no held-out accuracy.** Training-weight effect is +0.007 (i); a curator-usable
  source-median rule gives +0.001 (ii). All of the pre-registered −0.137 was the leak.
- **M6-2's remaining gain is M2's removal set.** M2 itself (u⁰, no leak) is −0.061 vs M1 pre-registered. It
  fails the lines leg (−0.38% kept), as before.
- **M3b-T is worse than M1 once its weights can't see the held-out line.** Its pre-registered edge came from
  the same leak.
- **Leak scope.** The same leak affects every pre-registered score whose u was fitted with the held-out line
  in the network: M0 (MARVEL 4-optimised u), M6-x, M3b-x, M5 (source scales; weaker, source-level). Only the
  methods that score with u⁰ are clean: M1, M2, M3a-x, M4 and their +R variants. The pre-registered tables
  stay as computed; read their inflation rows with this in mind.

## Recommendation for marvel-network on C2H4 v2 (advisory)

**Pre-registered pick: M6-1. Corrected post hoc: greedy M1 at k = 3, removals as flags; the replay is
optional bookkeeping, not an accuracy gain.**
1. Greedy |r|/u (M1): remove the max-z line and re-solve until max z ≤ 3.
2. Report the removals as a ranked flag list (removal order) for curator review.
3. If MARVEL-style conservative uncertainties are wanted, run the MARVEL 4 replay afterwards. The nested
   check shows it neither helps nor hurts held-out prediction (+0.001 nats), so it is a reporting
   convention.

**What it does on C2H4 v2 (k = 3; NLL leak-free, nested version (ii)):**

| | M0 curators | **M1** | M6-1 | M3b-T |
|---|---|---|---|---|
| lines kept | 0.9865 (679 removed) | 0.9368 (3,180) | 0.9368 (3,180) | 0.9517 (2,429) |
| validated levels | 0.7700 | 0.7519 | 0.7519 | 0.7525 |
| held-out NLL, nested (ii) | −5.071 (weak stand-in) | **−6.493** | −6.495 | −6.311 |
| common-set NLL, nested (ii) | −6.436 | **−6.560** | −6.563 | −6.535 |
| pre-registered (leaky) NLL | −5.550 | −6.493 | −6.671 | −6.529 |
| removed ∩ curator negations (precision / recall) | 1 / 1 | 0.10 / 0.49 | 0.10 / 0.49 | 0.15 / 0.53 |
| max z on own network | 14.4 | 3.0 | 1.0 | 2.4 |

**Reading for C2H4:**
- **Greedy M1 predicts held-out lines better than the curated network.** On the lines both keep it is
  −6.56 vs −6.44, against the M0 stand-in (a weak proxy for MARVEL 4's own optimisation). It removes 4.7× as many lines (6.3% vs 1.35%) and validates
  1.8 pp fewer levels.
- **The curated v1 network is not k-consistent at its own uncertainties** (max z 14.4).
- **Only 10% of M1's removals are curator negations.** The other ~2,850 are lines the curators kept, mostly
  with inflated uncertainty (13,589 inflated lines).
- **The real policy question is remove vs inflate.** The nested check says inflation buys no held-out
  accuracy, so the choice turns on lines and levels kept (curators win) against consistency and prediction
  (greedy wins).

**Recommendation (advisory; flags, not edits):**
- **Flag list.** Run M1 at k = 3 on C2H4 v2 and use its 3,180 removals, ranked, as a flag list. Review
  against the curated decisions, starting with the ~2,850 lines M1 removes and the curators kept. Include
  the curated network's max z of 14.4 in the flags.
- **Don't auto-remove** beyond what curators confirm.
- **k = 3.** k = 2 removes 4,489 C2H4 lines; k = 5 removes 2,069.
- **Not worth adding:** studentized greedy (M2), L1 (M4), Huber/Tukey variants, re-admission (+R) or
  Helmert source scales. None beats M1 on lines without losing accuracy. M3b-T also needs a weight floor
  before any further use.
- **Untested next steps:**
  - a sound exact check (tight per-line big-M or indicator constraints) plus a multi-start greedy on VO-like
    runs, where greedy was beaten by up to 62% in lines removed;
  - leak-free scoring for M5 and the remaining inflate rows, if inflation is revisited.

## Threats to validity

- **13 groups; CO2 is one group of 6 runs.** CIs are wide on everything involving M0.
- **Own-set NLL rewards removal**; the common-set NLL controls it.
- **Held-out u leak for inflating methods** (found post hoc; see the nested check). It overturned the
  pre-registered winner.
- **1-column runs may carry optimised u⁰** (CN, AlO, HOCl, CaOH, CO 13C-16O), which favours inflate-only
  methods there.
- **The MILP ignores connectivity and asks only for the existence of energies within k·u⁰**, not for the LS
  solution to meet k (see post hoc).
