"""Blind C2H4 test, step D: score frozen predictions against the released key.

Run: uv run python scripts/c2h4_blind_score.py <overrides.tsv>

Implements reports/2026-10-29_c2h4_blind/PREREGISTRATION.md. Checks the frozen
prediction hashes first and refuses to score if any file changed.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).parents[1]
PRED = ROOT / "data" / "C2H4_v1" / "predictions"
N_BOOT = 1000


def check_hashes():
    for line in (PRED / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        path = ROOT / "data" / name if name.endswith(".pt") else PRED / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"{name} changed since freeze"


def load_lines(name):
    rows = (PRED / f"{name}_lines.tsv").read_text().splitlines()
    head = rows[0].split("\t")
    data = [r.split("\t") for r in rows[1:]]
    out = {h: [d[k] for d in data] for k, h in enumerate(head)}
    for h in head[3:]:
        out[h] = np.array(out[h], dtype=float)
    return out


def ap(labels, scores):
    """Average precision, ties broken by line order (stable sort)."""
    hits = labels[np.argsort(-scores, kind="stable")]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / hits.sum())


def recall_at(labels, scores, k):
    top = np.argsort(-scores, kind="stable")[:k]
    return float(labels[top].sum() / labels.sum())


def auroc(labels, scores):
    r = rankdata(scores)
    n1 = labels.sum()
    n0 = len(labels) - n1
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def outlier_block(labels, scores):
    """AP, recall@k (k = #positives), recall in top 1% for each score."""
    k = int(labels.sum())
    top = math.ceil(0.01 * len(labels))
    return {name: (ap(labels, s), recall_at(labels, s, k), recall_at(labels, s, top))
            for name, s in scores.items()}


def bootstrap(labels, scores, metric, pairs, seed=0):
    """95% CIs of metric per score and of metric(a) - metric(b) per pair."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    draws = {k: [] for k in list(scores) + [f"{a}-{b}" for a, b in pairs]}
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if labels[i].sum() in (0, n):
            continue
        m = {k: metric(labels[i], s[i]) for k, s in scores.items()}
        for k in scores:
            draws[k].append(m[k])
        for a, b in pairs:
            draws[f"{a}-{b}"].append(m[a] - m[b])
    return {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in draws.items()}


def demo():
    lab = np.array([1, 0, 1, 0, 0], dtype=bool)
    assert ap(lab, np.array([5, 4, 3, 2, 1.0])) == (1 + 2 / 3) / 2
    assert auroc(lab, np.array([5, 4, 3, 2, 1.0])) == 5 / 6
    assert recall_at(lab, np.array([5, 4, 3, 2, 1.0]), 2) == 0.5


C2H4 = ROOT / "data" / "C2H4_v1"
N_ITER = 641
OUTLIER_SCORES = ["gnn_outlier_logit", "resid_over_unc", "studentized", "iterative", "random"]
UNC_SCORES = ["gnn_sigma_over_unc", "gnn_line_sigma", "resid_over_unc", "studentized", "random"]
BLANK_UNC = {"10LoBoLoRo": 0.0005, "24PeLeSuRu": 0.005, "26PeLeSuRu": 0.0013,
             "18AsRa": 1e-3, "23MrMaReAr": 1e-3, "24BeBeReMa": 1e-3}  # pre-registered placeholders


def load_key(path):
    """overrides.tsv row types -> pre-registered sets (see report, key mapping)."""
    key = {"negate": {}, "set_unc": {}, "scale": {}, "assume_unc": {}, "exclude": {}}
    for row in Path(path).read_text().splitlines()[1:]:
        action, target, value, reason = row.split("\t")
        key[action][target] = (value, reason)
    return key


def n_categories(reason):
    cats = set()
    if "QAC" in reason: cats.add("QAC")
    if "QAB" in reason: cats.add("QAB")
    if "QAI" in reason or "QAM" in reason: cats.add("QAI/QAM")
    if "QAJ" in reason: cats.add("QAJ")
    if "QAU" in reason: cats.add("QAU")
    return cats


def unc_categories(reason):
    cats = set()
    if "QAS a'" in reason: cats.add("QAS a' (rule)")
    if "QAS 08RoBoVa" in reason: cats.add("QAS 08RoBoVa")
    if "QAR" in reason: cats.add("QAR")
    if "QAU" in reason: cats.add("QAU")
    return cats


def deliverable():
    out = {}
    for row in (C2H4 / "C2H4_12C2H4_trans.txt").read_text().splitlines():
        t = row.split()
        if len(t) > 5:
            out[t[-1]] = (float(t[0]), float(t[1]), float(t[2]))
    return out


def source_negations():
    ids = set()
    for f in (C2H4 / "input").glob("C2H4_12C2H4_*_qam.txt"):
        for row in f.read_text().splitlines()[1:]:
            t = row.split("\t")
            if float(t[0]) < 0.0:
                ids.add(t[32])
    return ids


def fmt_block(res):
    return "\n".join(f"    {k:18s} AP {a:.3f}  recall@k {r:.3f}  top1% {t:.3f}" for k, (a, r, t) in res.items())


def main(key_path):
    demo()
    check_hashes()
    key = load_key(key_path)
    L = load_lines("unedited")
    ids = np.array(L["id"])
    src = np.array([i.split(".")[0] for i in ids])
    main_ = L["in_main"] == 1
    it = L["iterative_rank"]
    scores = {"gnn_outlier_logit": L["gnn_outlier_logit"], "resid_over_unc": L["resid_over_unc"],
              "studentized": L["studentized"],
              "iterative": np.where(it < N_ITER, -it, -float(N_ITER)),  # never-removed tied at bottom
              "random": L["random"]}
    neg_ids = set(key["negate"])
    S_ids = source_negations()
    excluded = set(key["exclude"])
    inN = np.isin(ids, list(neg_ids))
    inS = np.isin(ids, list(S_ids))
    inX = np.isin(src, list(excluded))
    negatives = main_ & ~inN & ~inS & ~inX
    print(f"key: negate {len(neg_ids)}, set_unc {len(key['set_unc'])}, scale {len(key['scale'])}, "
          f"assume_unc {len(key['assume_unc'])}, exclude {sorted(excluded)}")
    absent = [x for x in excluded if not (src == x).any()]
    print(f"excluded sources absent from unedited inputs (row fits no set, not scored): {absent}")
    print(f"N {inN.sum()} (in ground comp {(inN & main_).sum()}), S {inS.sum()} ({(inS & main_).sum()}), "
          f"X {inX.sum()} ({(inX & main_).sum()}), negatives {negatives.sum()}")
    missing = neg_ids - set(ids)
    print(f"negate targets not found in unedited lines: {len(missing)}")

    results = {}

    def score_set(name, pos, boot=False):
        pool = negatives | (pos & main_)
        lab = pos[pool]
        if lab.sum() == 0:
            print(f"  {name}: no positives in ground component"); return
        sc = {k: v[pool] for k, v in scores.items()}
        res = outlier_block(lab, sc)
        results[name] = {"n_pos": int(lab.sum()), "pool": int(pool.sum()), "metrics": res}
        print(f"  {name}: {int(lab.sum())} positives in pool of {int(pool.sum())} "
              f"(random AP expect {lab.mean():.4f})")
        print(fmt_block(res))
        lev = L["leverage"][pool]
        bridge = lab & (lev >= 0.99)
        if bridge.any():
            k = int(lab.sum())
            rec = {kk: float(np.isin(np.where(bridge)[0], np.argsort(-s, kind="stable")[:k]).mean())
                   for kk, s in sc.items()}
            print(f"    h>=0.99 positives {int(bridge.sum())}: recall@k " +
                  "  ".join(f"{kk} {v:.2f}" for kk, v in rec.items()))
        if boot:
            pairs = [("gnn_outlier_logit", b) for b in ["resid_over_unc", "studentized", "iterative"]]
            best = max(["resid_over_unc", "studentized", "iterative"], key=lambda b: res[b][0])
            ci = bootstrap(lab, sc, ap, pairs)
            results[name]["ci"] = ci
            results[name]["best_baseline"] = best
            for kk, (lo, hi) in ci.items():
                print(f"    95% CI {kk:40s} [{lo:+.3f}, {hi:+.3f}]")
            print(f"    best baseline on full pool: {best}")

    print("\n== OUTLIER (unedited graph) ==")
    print("PRIMARY: N pooled")
    score_set("N pooled", inN, boot=True)
    cats = {i: n_categories(key["negate"][i][1]) for i in neg_ids}
    for c in ["QAB", "QAC", "QAI/QAM", "QAJ", "QAU"]:
        score_set(f"N {c}", np.isin(ids, [i for i, cs in cats.items() if c in cs]))
    other = [i for i, cs in cats.items() if not cs]
    print(f"  negate rows with no pre-registered category: {len(other)}")
    print("S pooled")
    score_set("S pooled", inS, boot=True)
    for s_ in sorted({i.split('.')[0] for i in S_ids}):
        score_set(f"S {s_}", inS & (src == s_))
    score_set("X (excluded sources)", inX)

    print("\n== UNCERTAINTY (unedited graph) ==")
    dl = deliverable()
    unc0 = L["unc"]
    factor = np.array([dl[i][2] / u if i in dl else np.nan for i, u in zip(ids, unc0)])
    scaled = set(key["scale"])
    assumed = set(key["assume_unc"])
    has_set = np.isin(ids, list(key["set_unc"]))
    eligible = main_ & ~inN & ~inS & ~inX
    in_scaled = np.isin(src, list(scaled))
    pos = eligible & has_set
    neg = eligible & ~has_set & ~in_scaled
    pool = pos | neg
    lab = pos[pool]
    usc = {k: L[k][pool] for k in UNC_SCORES}
    print(f"(a) line-level: {int(lab.sum())} set_unc positives, {int((~lab).sum())} negatives "
          f"(set_unc rows outside eligible lines: {len(key['set_unc']) - int(pos.sum())})")
    au = {k: auroc(lab, v) for k, v in usc.items()}
    ci = bootstrap(lab, usc, auroc, [("gnn_sigma_over_unc", b) for b in ["resid_over_unc", "studentized"]])
    results["unc_a"] = {"auroc": au, "ci": ci, "n_pos": int(lab.sum()), "n_neg": int((~lab).sum())}
    print("  PRIMARY AUROC (gnn_sigma_over_unc) and baselines:")
    for k in UNC_SCORES:
        print(f"    {k:20s} AUROC {au[k]:.3f}  95% CI [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    for k in ["gnn_sigma_over_unc-resid_over_unc", "gnn_sigma_over_unc-studentized"]:
        print(f"    diff {k:40s} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]")
    fpos = factor[pool][lab]
    from scipy.stats import spearmanr
    print("  Spearman vs applied factor among positives: " +
          "  ".join(f"{k} {spearmanr(v[lab], fpos).statistic:+.3f}" for k, v in usc.items()))
    ucats = {i: unc_categories(key["set_unc"][i][1]) for i in key["set_unc"]}
    pid = ids[pool]
    results["unc_a_cat"] = {}
    for c in ["QAS a' (rule)", "QAS 08RoBoVa", "QAR", "QAU"]:
        cl = np.isin(pid, [i for i, cs in ucats.items() if c in cs]) & lab
        sub = cl | ~lab
        a = {k: auroc(cl[sub], v[sub]) for k, v in usc.items()}
        results["unc_a_cat"][c] = {"n": int(cl.sum()), "auroc": a}
        print(f"    {c:16s} n={int(cl.sum()):3d}  " + "  ".join(f"{k} {v:.3f}" for k, v in a.items()))
    print(f"    set_unc rows with no pre-registered category: {sum(1 for cs in ucats.values() if not cs)}")

    # applied factor vs key
    fe = factor[eligible]
    se = src[eligible]
    ie = ids[eligible]
    hs = has_set[eligible]
    near1 = np.abs(fe - 1.0) < 2e-3  # unedited unc is stored to 4 significant figures
    expect = np.array([float(key["scale"][s_][0]) if s_ in scaled else np.nan for s_ in se])
    d1 = hs & near1
    d2 = ~hs & ~near1 & ~np.isin(se, list(scaled | assumed))
    d3 = ~hs & np.isin(se, list(scaled)) & (np.abs(fe / expect - 1.0) > 2e-3)
    print(f"  factor vs key: set_unc rows with factor ~1: {int(d1.sum())}; "
          f"factor != 1 without set_unc/scale/assume: {int(d2.sum())}; "
          f"scaled-source lines whose factor != scale: {int(d3.sum())}")
    for name, m in [("set_unc factor~1", d1), ("unexplained factor", d2), ("scale mismatch", d3)]:
        ex = [f"{i} x{f:.3g}" for i, f in zip(ie[m][:5], fe[m][:5])]
        if ex:
            print(f"    e.g. {name}: {', '.join(ex)}")
    results["unc_disagree"] = {"setunc_factor1": int(d1.sum()), "unexplained": int(d2.sum()),
                               "scale_mismatch": int(d3.sum())}

    print("(b) source-level (eligible lines):")
    rows = []
    for s_ in sorted(set(src[eligible])):
        m = eligible & (src == s_)
        f = (float(key["scale"][s_][0]) if s_ in scaled else
             float(key["assume_unc"][s_][0]) / BLANK_UNC[s_] if s_ in assumed else 1.0)
        kind = "scale" if s_ in scaled else "assume" if s_ in assumed else "-"
        rows.append((s_, kind, f, float(np.median(L["gnn_sigma_over_unc"][m])),
                     float(np.median(L["resid_over_unc"][m])), int(m.sum())))
    for r in rows:
        print(f"    {r[0]:12s} {r[1]:6s} factor {r[2]:7.3g}  median gnn_sigma/unc {r[3]:.3g}  "
              f"median resid/unc {r[4]:.3g}  n={r[5]}")
    sc_rows = [r for r in rows if r[1] == "scale"]
    rho = spearmanr([r[2] for r in sc_rows], [r[3] for r in sc_rows]).statistic
    print(f"  Spearman over {len(sc_rows)} scaled sources (factor vs median gnn_sigma/unc): {rho:+.3f}")
    results["unc_b"] = {"rows": rows, "spearman_scaled": rho}

    print("\n== UNCERTAINTY (edited graph vs states.txt) ==")
    st = {}
    for row in (C2H4 / "C2H4_12C2H4_states.txt").read_text().splitlines():
        t = row.split("\t")
        st[t[0]] = (float(t[2]), int(t[3]), t[4])
    lv = (PRED / "edited_levels.tsv").read_text().splitlines()[1:]
    a = [r.split("\t") for r in lv]
    a = [(x[0], float(x[2])) + st[x[0]] for x in a if st[x[0]][0] > 0]
    g = np.array([x[1] for x in a]); u = np.array([x[2] for x in a])
    status = np.array([x[4] for x in a])
    print(f"  levels {len(a)}: Spearman(gnn sigma, states unc) {spearmanr(g, u).statistic:+.3f}")
    for s_ in ["validated", "tree"]:
        m = status == s_
        print(f"    {s_:9s} n={int(m.sum())}: {spearmanr(g[m], u[m]).statistic:+.3f}")
    lr = np.log10(g / u)
    print(f"  median log10(gnn/states) {np.median(lr):+.2f}")
    for lab_, order in [("GNN >> states", np.argsort(-lr)[:20]), ("GNN << states", np.argsort(lr)[:20])]:
        print(f"  top 20 {lab_}:")
        for k in order:
            print(f"    {a[k][0]:40s} gnn {a[k][1]:.2e} states {a[k][2]:.2e} deg {a[k][3]} {a[k][4]}")
    results["edited"] = {"spearman": spearmanr(g, u).statistic}

    E = load_lines("edited")
    em = E["in_main"] == 1
    top = np.argsort(-np.where(em, E["gnn_outlier_logit"], -np.inf), kind="stable")[:50]
    adv = ["id\tupper\tlower\tfreq\tunc\tgnn_outlier_logit\tresid_over_unc\tleverage\tstudentized"]
    for k in top:
        adv.append("\t".join([E["id"][k], E["upper"][k], E["lower"][k], f"{E['freq'][k]:.6f}",
                               f"{E['unc'][k]:.3e}"] +
                              [f"{E[c][k]:.4g}" for c in ["gnn_outlier_logit", "resid_over_unc",
                                                          "leverage", "studentized"]]))
    out = ROOT / "reports" / "2026-10-29_c2h4_blind" / "advisory_top50_edited.tsv"
    out.write_text("\n".join(adv) + "\n")
    print(f"\nadvisory: top 50 edited-graph lines -> {out.name}")
    import json
    (ROOT / "reports" / "2026-10-29_c2h4_blind" / "results.json").write_text(
        json.dumps(results, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
