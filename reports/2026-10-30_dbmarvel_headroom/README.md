# db_MARVEL headroom study (step 0)

**Task:** vaultagent-26, approved by the user. No GNN training or inference.
**Question:** on real curated MARVEL datasets, how much of the curators' outlier and uncertainty work does
the no-model pipeline already recover? The no-model pipeline is the iterative negate-worst-and-re-solve loop
plus cheap deterministic checks.
**Status:** phase 1 (label audit) and phase 2 (headroom) done. The answer and recommendation are at the end.

## Phase 1: label audit

- **Data.** db_MARVEL at 010b888 (clean tree) was copied to `data/dbmarvel/` (gitignored). The copy holds the
  manifests, the raw run files (without zip/pdf/tar.gz), and db_MARVEL's `loader/parse.py` and
  `loader/manifest.py`; hashes are in `SHA256SUMS`.
- **db_MARVEL was not running.** The DB is down and none of its files were written.
- **Manifests to JSON.** `manifests.json` was produced once by db_MARVEL's own `load_manifest`, run with
  db_MARVEL's interpreter (`python -B -I`, so nothing was written there). This avoids adding PyYAML here.
- **Script:** `scripts/dbmarvel_audit.py`; table: `audit.tsv` (one row per run, 33 rows).

### Per run

The table has 21 publication folders, 31 input runs, plus 12C2H4 v1.
- PN's `output_only` duplicate is skipped.
- N2O (24TeFuYu) has no manifest. Its layout was read off the file.

Column key:
- **pub ok:** share of published levels the Python port reproduces within their published unc, re-solving
  the edited largest component.
- **neg (main):** negated lines, and how many sit in the largest component.
- **N:** network negations in the largest component, excluding sources whose every line is negated.
- **U:** raised-uncertainty lines (used > original); in brackets, how many have used ≈ the final residual
  (|r| > 0.9·used).

| run | molecule | unc cols | lines | levels (largest) | pub ok | neg (main) | N | U (at resid) | usable |
|---|---|---|---|---|---|---|---|---|---|
| 09MiTaPu | H2O 1H-2H-18O | 1 | 970 | 615 (297) | 0.49 | 0 | 0 | — | no: not a MARVEL run, no labels |
| 20SyMc | CN | 1 | 40,516 | 8,480 (8,093) | 1.000 | 183 (183) | 183 | — | outlier |
| 20YiOwTe | CaOH | 1 | 3,240 | 1,955 (1,955) | 0.948 | 16 (16) | 16 | — | outlier |
| 21BoShYuTe | AlO | 1 | 23,972 | 6,588 (6,525) | 1.000 | 1,499 (1,499) | 1,499 | — | outlier |
| 22BoAkHoYuTe | VO fine | 2 | 9,344 | 5,113 (4,813) | 1.000 | 54 (54) | 54 | 1,337 (1,333) | both |
| 22BoAkHoYuTe | VO hfs | 2 | 8,082 | 6,015 (5,703) | 0.999 | 41 (41) | 41 | 241 (240) | both |
| 22RaFuFu | HOCl | 1 | 21,464 | 6,374 (6,370) | 1.000 | 147 (147) | 147 | — | outlier |
| 23BoMiZoPo | H3+ | 2 | 1,719 | 796 (705) | 1.000 | 63 (60) | 60 | 113 (107) | both |
| 23BoMiZoPo | H2D+ | 2 | 210 | 123 (109) | 1.000 | 2 (2) | 2 | 19 (19) | uncertainty |
| 23BoMiZoPo | D2H+ | 2 | 210 | 132 (115) | 0.009 | 0 | 0 | 0 | no: no labels; units wrong |
| 23BrYuTe | SO | 2 (identical) | 50,105 | 9,469 (8,695) | 0.997 | 1,134 (388) | 384 | 0 | outlier |
| 24AlIbFu | CO2 628 | 2 | 33,657 | 9,090 (8,787) | 1.000 | 95 (95) | 95 | 7,936 (5,010) | both |
| 24AzAzAb | CO2 828 | 2 | 11,654 | 4,191 (3,923) | 1.000 | 0 | 0 | 1,361 (1,180) | uncertainty |
| 24AzAzAb | CO2 728 | 2 | 11,339 | 4,546 (4,318) | 1.000 | 0 | 0 | 1,219 (1,164) | uncertainty |
| 24AzAzAb | CO2 838 | 2 | 2,426 | 1,170 (1,058) | 1.000 | 0 | 0 | 229 (192) | uncertainty |
| 24IbAlFu | CO2 636 | 2 | 20,798 | 6,538 (6,320) | 1.000 | 37 (29) | 29 | 6,033 (4,030); 1,309 lowered | both |
| 24McPeSyBo | TiO | 2 (identical) | 62,935 | 12,510 (12,300) | 1.000 | 1,426 (1,416) | 1,416 | 0 | outlier |
| 24SeElYu | PN | 2 | 5,941 | 1,689 (1,662) | 0.985 | 3,077 (3,077) | 1,695 | 252 (194) | both |
| 25AzAlAb | CO2 626 | 2 | 44,827 | 8,917 (8,294) | 1.000 | 368 (349) | 284 | 11,283 (9,483) | both |
| 25AzAlTe | CO2 737 | 2 | 1,831 | 1,132 (877) | 1.000 | 0 | 0 | 205 (170) | uncertainty |
| 25AzAlTe | CO2 738 | 2 | 2,058 | 1,185 (945) | 1.000 | 0 | 0 | 290 (258) | uncertainty |
| 25AzTeYu | CO2 638 | 2 | 12,362 | 4,405 (3,975) | 1.000 | 21 (21) | 21 | 2,513 (2,219) | both |
| 25MaAzTe | CO2 637 | 2 | 5,053 | 2,188 (1,960) | 1.000 | 0 | 0 | 810 (600) | uncertainty |
| 25MaAzTe | CO2 727 | 2 | 10,256 | 3,504 (3,322) | 1.000 | 90 (85) | 85 | 2,406 (1,965) | both |
| 25MaElAb | CO 12C-16O | 2 | 19,399 | 2,500 (2,295) | 1.000 | 12 (12) | 12 | 3,783 (3,768) | both |
| 25ObAzTe | CO2 627 | 2 | 16,224 | 5,281 (5,040) | 1.000 | 103 (103) | 103 | 3,242 (2,611) | both |
| 26GrDaPo | CO 12C-17O | 1 | 90 | 85 (34) | 1.000 | 1 (1) | 1 | — | no: < 10 |
| 26GrDaPo | CO 12C-18O | 1 | 1,461 | 766 (498) | 1.000 | 7 (7) | 0 | — | no: < 10 (all 7 in one whole source) |
| 26GrDaPo | CO 13C-16O | 1 | 3,731 | 1,216 (863) | 1.000 | 15 (15) | 15 | — | outlier |
| 26GrDaPo | CO 13C-17O | 1 | 58 | 45 (45) | 0.956 | 0 | 0 | — | no |
| 26GrDaPo | CO 13C-18O | 1 | 728 | 517 (345) | 1.000 | 0 | 0 | — | no |
| 24TeFuYu | N2O 446 | 2 | 67,930 | 18,345 (17,662) | 1.000 | 635 (634) | 632 | 15,615 (13,429) | both |
| MARVEL_network 0b56238 | C2H4 | key | 50,611 | 12,109 (11,687) | — | 641 + 58AlPl | 625 | 827 set_unc (curator) | both |

**Totals.**
- Outlier: 21 runs in 13 molecule groups, 7,396 network negations. AlO, TiO and PN make up 4,610 of
  them, which is why the pooling weights by molecule.
- Uncertainty: 20 runs in 7 groups.

### Label schema findings

1. **No removal reasons exist in db_MARVEL.** The `removed` and `removed_reason` columns are solve-state fields
   that the loader strips, and no raw file carries them.
   - The only per-line annotations are a few `-relab`, `-typo` and `BLD` notes (CO2, H2O), and `-X0.n` on
     3 CaOH PGOPHER lines.
   - So, outside C2H4, the only reason proxy is **whole-source exclusion**: a source with every line
     negated. That covers 4 sources in SO (600 lines), PN's 81CoPrxx (1,382), CO2 626's 00TaPeTeLe (65),
     CO 12C-18O (7) and N2O (2).
   - Rule-shaped vs outlier-shaped splits exist only for C2H4.
2. **The "uncertainty raised" label is mostly algorithmic.**
   - In every 2-column run, the edited re-solve with the used uncertainties is consistent: no unraised line
     has |r| > used.
   - Most raised lines sit at used ≈ |r|: VO 99.7%, CO 99.6%, H3+ 95%, CO2 63–94%, N2O 86%, PN 77%.
   - This is MARVEL 4's automatic uncertainty optimisation (our own `Transition.unc` docstring already calls
     it the "optimized uncertainty"), not a curator decision.
   - Factors are continuous (1.02, 1.11, …), not round numbers. The non-at-residual remainder includes some
     round values that look like curator settings: CO2 626 has 117 lines set to exactly 0.05 cm-1, and CO2
     627 has 30 lines at ×10.
   - C2H4's key `set_unc` is the only clean curator uncertainty label.
3. **Two-column files where both columns are identical:** SO and TiO. They have no uncertainty label; TiO's
   edited re-solve leaves 817 lines with |r| > unc.
4. **One-column runs** (CN, CaOH, AlO, HOCl, CO 26GrDaPo): no original-vs-used split. CN and AlO re-solve with
   0 inconsistent lines, which suggests the single column is already optimised (see the pre-registration's
   threats to validity).
5. **CO2 636 lowers 1,309 uncertainties** (factors 0.70 / 0.75). The pre-registration drops them from the
   uncertainty pool.

### Parsing problems found (db_MARVEL side; reported, not fixed there)

- **CaOH (20YiOwTe).** The manifest's `numeric_field_widths: [14, 12]` mis-slices the 675 rows whose unc field
  is not right-aligned within 12 characters.
  - On those rows the unc is lost (→ NULL) and the first character of the upper label is clipped
    (`muA2Sigma` → `uA2Sigma`), which creates spurious levels.
  - Only 206 rows are actually glued (`15961.7230.0246092`). The local copy splits only those
    (`fixed/20YiOwTe_*`); after that, 0 lines have a missing unc.
  - db_MARVEL's loaded CaOH run is affected.
- **D2H+ (23BoMiZoPo).** The manifest declares `05AmHi` and `08AsRiMuWi` as MHz, but their values are cm-1:
  inference against the published levels gives cm-1, and the re-solve reproduces only 0.9% of levels.
  db_MARVEL's loaded D2H+ run is affected.
- **N2O (24TeFuYu).** No manifest and no segment file. All 91 source units are inferred from the published
  levels; 1,030 lines are MHz/kHz, and the re-solve then reproduces 100% of levels. The QN names are
  unverified.
- **Duplicate tags:** 0 apart from CaOH (1). Self-loops: 0.

## Phase 2: headroom (run 2026-10-30)

**Pre-registration** sha256 `d84f8a6d6924b3d8d6f4126e24b2896499da05d46a95b28085db7ed83571b75c`. This is the
revised version: the user's change, the second partition with the label-free loop, was folded in before any
score was computed.
- **Scripts:** `scripts/dbmarvel_headroom.py`, and `scripts/dbmarvel_posthoc.py` (post hoc).
- **Outputs:** `results.json`, `results.txt`, `posthoc.json`, `posthoc_output.txt`.
- **Cache:** line scores in `data/dbmarvel/scores/`.

### Deviations from the pre-registration (all forced, all applied to every run)

1. **Leverage uses unc floored at UNC_FLOOR (1e-6 cm-1).**
   - PN's line uncertainties span 4.6e-10 to 0.2 cm-1, a weight ratio of about 1e17. That is past double
     precision, so the LDL^T factor fails its own Σh = n − 1 check (1568 vs 1661).
   - The floor affects leverage only, which feeds the bridge flag and the studentized score. Residuals,
     loops and replays are unfloored.
   - Runs with lines under the floor: CO2 626 (1068), N2O (384), SO (316), CO2 828/838 (about 220 each),
     C2H4 (116), and a few elsewhere.
2. **AP and recall are also reported tie-aware.**
   - The pre-registered "ties broken by line order" is invalid for tied scores. The binary rule score gets
     AP 0.90–1.00 on HOCl, CO2 628/636, TiO, PN and N2O while firing on 0 positives, because negated lines
     sit early in those files.
   - The same artifact inflates the loop's tied bottom block. AP_ties scores a tie block as one threshold
     step (expected precision under random order within it); recall_ties gives pro-rata credit to the tie
     block at the cut-off. Both are self-checked.
   - Both versions are in `results.txt`. **The tie-aware numbers are the ones quoted below.**
   - The headline B/L/R/H partition doesn't depend on AP and is unaffected.
   - The effect on C2H4 is small: (iv) AP 0.440 → 0.433. The blind test's (iv) figure of 0.425 is therefore
     not materially inflated.
3. **Details the pre-registration left open:**
   - In P, non-negated lines cut off from the pruned graph get factor 1.
   - R2 doesn't apply to AlO and PN: their `ef` column holds +/− and ±1 tokens, which the pre-registered
     token rule skips.

### Headline: partition of the real network negations N (pooled, molecule-weighted, 13 groups, 95% cluster-bootstrap CI)

| bucket | L from (iv): K = #negations | L′ from (iv′): label-free stop |
|---|---|---|
| B bridge (h ≥ 0.99) | 0.066 [0.028, 0.117] | 0.066 [0.028, 0.117] |
| L caught by the loop | 0.655 [0.559, 0.758] | 0.643 [0.513, 0.764] |
| R caught only by a generic rule | 0.014 [0.000, 0.040] | 0.014 [0.000, 0.039] |
| **H headroom** | **0.266 [0.188, 0.337]** | **0.278 [0.174, 0.390]** |
| unweighted, per line (B / L / R / H) | 0.036 / 0.593 / 0.015 / 0.356 | 0.036 / 0.555 / 0.015 / 0.393 |
| groups with H ≥ 0.25 | 7 of 13 | 7 of 13 |
| pre-registered verdict | judgement call | judgement call |

The two partitions agree on the verdict.

**H per group, (iv) / (iv′):**

| group | H (iv) / (iv′) |
|---|---|
| PN | 0.51 / 0.18 |
| TiO | 0.43 / 0.66 |
| N2O | 0.40 / 0.08 |
| CO2 | 0.37 / 0.27 |
| HOCl | 0.33 / 0.33 |
| VO | 0.27 / 0.04 |
| C2H4 | 0.27 / 0.25 |
| CN | 0.25 / 0.44 |
| SO | 0.23 / 0.46 |
| H3+ | 0.23 / 0.20 |
| AlO | 0.13 / 0.60 |
| CO | 0.04 / 0.04 |
| CaOH | 0.00 / 0.06 |

- **The 3K cap binds in 10 runs:** VO fine, CO2 628/636/626/638/727/627, CO 12C-16O, N2O and C2H4. The
  max |r|/unc ≤ 3 stop is never reached there, so (iv′) is label-free only where it stops on its own: CN,
  CaOH, AlO, VO hfs, HOCl, H3+, SO, TiO, PN and CO 13C-16O.
- **Precision of the loop's K removals:**
  - median 0.71;
  - CO2 626 0.14, C2H4 0.45, HOCl 0.37;
  - CaOH and CO 13C-16O 1.0.

### Outlier AP (tie-aware, pooled, molecule-weighted)

| score | AP | 95% CI |
|---|---|---|
| (iv) loop, K from the labels | **0.595** | [0.496, 0.707] |
| (iv′) label-free loop | 0.544 | [0.446, 0.645] |
| (i) one-pass \|r\|/unc | 0.407 | [0.299, 0.529] |
| (ii) studentized | 0.356 | [0.267, 0.448] |
| (R) generic rules | 0.044 | [0.010, 0.099] |
| random (prevalence) | 0.043 | [0.008, 0.098] |

- At group level the loop beats the one-pass residual in 12 of 13 groups. The exception is HOCl (0.37 vs
  0.66). Within CO2, run 727 also favours the residual (0.46 vs 0.61), and on 626 every score is near random
  (AP ≤ 0.03).
- The generic rules are no better than random overall. Their only real catch is C2H4: 63 of the 75 QAB
  (symmetry-block) negations and 13 of 59 QAI/QAM. But on C2H4 they also flag 5,241 non-negated lines, and
  on AlO 2,428.

**C2H4 by key category** (n; B / L(iv) / R): QAB 75: 0 / 6 / 63 · QAI/QAM 59: 37 / 2 / 13 · QAC 350:
8 / 205 / 23 · QAJ 63: 0 / 41 / 1 · QAU 5: 0 / 5 / 0 · source negations 81: 7 / 40 / 25. Same shape as the
blind test: the outlier-shaped categories are what the loop finds.

### What the headroom lines look like (post hoc, `posthoc_output.txt`)

Of the 2,633 H lines (iv, all runs), 1,142 (43%) have one-pass |r|/unc < 3: no residual-type feature can see
them. The other 57% are visible but outranked; PN alone is 835 of those.
- **Invisible to residuals** (|r|/unc < 3 for most H lines):
  - TiO: 96% of 602 lines;
  - AlO: 71% of 190; CN: 67%; CO2 626: 62%;
  - SO: 72% sit in sources with at least half their lines negated.
  - These look like curation for reasons outside the network: superseded or duplicate measurements, or
    partially excluded sources.
- **Visible but outranked** (|r|/unc ≥ 3 for nearly all H lines): PN (96% of 870), HOCl (all 48), VO, CO2
  628/638/727, H3+.
  - The loop spent its budget on non-negated high-residual lines, the ones curators kept and handled by
    raising their uncertainty.
  - So the open decision here is "negate or inflate?" among lines that are already flagged.

### Uncertainty (7 groups, 20 runs)

Pooled AUROC for "uncertainty raised":

| score | AUROC | 95% CI | ρ with applied factor |
|---|---|---|---|
| **P** (loop, then MARVEL 4-style replay) | **0.887** | [0.783, 0.952] | 0.69 |
| P\* (replay on the curated graph) | 0.911 | [0.836, 0.961] | 0.78 |
| (i) \|r\|/unc | 0.819 | [0.726, 0.901] | 0.54 |
| (ii) studentized | 0.804 | | |
| per-source scatter | 0.672 | | 0.12 |
| 1/unc | 0.496 | | |

- **The pre-set "U is algorithmic" bar (AUROC ≥ 0.95 and ρ ≥ 0.8) is not met pooled.** The shortfall sits
  in 2 groups:
  - PN: P 0.59, ρ 0.02. The replay raises 2,554 of 2,843 lines, so PN's original uncertainties are far too
    small throughout.
  - C2H4: 0.87, ρ 0.31. This is the only curator-decided label.
- Every other group is at P ≥ 0.91 and ρ ≥ 0.81 (CO 0.97, N2O 0.96, H3+ 0.96, CO2 0.95, VO 0.91). There,
  the label is effectively MARVEL's own optimisation, and replay recovers it.
- **On C2H4's curator label,** the no-model replay (0.870) beats the blind test's CO-trained GNN (0.739)
  and 1/unc (0.739). That comparison is post hoc.

## Answer and recommendation

**Real network negations (pooled, molecule-weighted, labelled K):**
- (a) caught by the loop: 0.66 (label-free 0.64);
- (b) bridges: 0.07;
- caught only by a generic rule: 0.01;
- (c) **headroom: 0.27 [0.19, 0.34]** (label-free 0.28 [0.17, 0.39]).

**Uncertainty.** Replaying MARVEL 4's optimisation after the loop reaches AUROC 0.89 pooled, and ≥ 0.91 in
5 of 7 groups. That is the bar an uncertainty head must beat. It already beats the CO GNN on C2H4.

**Verdict.** Both partitions land in the pre-registered judgement band (0.20 ≤ H < 0.40, with 7 groups at
H ≥ 0.25). My call: **don't run step 1 as specified** (a real-label leave-one-molecule-out GNN on the
current residual and structure features). Run a cheaper step first. Reasons:
1. **About 43% of headroom is invisible to residuals** (|r|/unc < 3). The current GNN's inputs are
   residuals, leverages, uncertainties and structure, so it can't see those lines any better than the loop
   does. Recovering them needs other inputs: source metadata, duplicate and superseded detection, and
   per-source negation rates. These are cheap, deterministic or tabular signals, not graph learning.
2. **The rest of headroom is a ranking question among already-flagged lines:** negate or inflate? That
   target is narrow and well posed, and a small tabular classifier (leave-one-molecule-out, against the
   loop) tests it far more cheaply than a GNN.
   - Candidate features: |r|/unc, studentized, |r|/source scatter, source negation rate, unc relative to the
     source median, degree.
   - If that classifier can't beat the loop, a GNN on the same signals won't either. If it can, a loop + GNN
     hybrid becomes a targeted follow-up rather than a broad bet.
3. **Uncertainty has no learnable target outside C2H4** (one molecule, 827 labels). In every other group
   replay already reproduces the label.

H is an upper bound: some headroom lines were negated on evidence that no network model can see.
