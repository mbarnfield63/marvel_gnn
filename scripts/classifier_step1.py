"""Step 1': can a line classifier on molecule-invariant features beat the
iterative loop at ranking real network negations, leave-one-molecule-out?

Run: uv run python scripts/classifier_step1.py

Pre-registered in reports/2026-10-30_classifier/PREREGISTRATION.md. Reuses
the headroom study's runs, labels and cached scores (data/dbmarvel/scores/);
adds the label-free loop L5 (cached as *_L5.npz) and the features below.
Writes reports/2026-10-30_classifier/{results.json,results.txt}.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).parent))
from c2h4_blind_score import auroc  # noqa: E402
from dbmarvel_headroom import (CACHE, ROOT, ap_ties, cached, largest, recall_ties,  # noqa: E402
                               records, resid)

OUT = ROOT / "reports" / "2026-10-30_classifier"
N_BOOT = 1000
NEG_FRAC = 0.2
FEATURES = ["log_z", "log_stud", "h", "loop_pos",                       # residual
            "log_deg_u", "log_deg_l",                                   # structure
            "src_flagged", "src_scatter", "src_log_size", "src_share", "unc_rel_src",  # source
            "has_dup", "n_dup_src", "best_other_ratio", "dup_discrep", "newer_dup",
            "src_year_pct", "year_missing"]                             # duplicate/superseded
FAMILY = {"residual": FEATURES[:4], "structure": FEATURES[4:6], "source": FEATURES[6:11],
          "dup": FEATURES[11:]}
MONOTONE = {"log_z": 1, "log_stud": 1, "loop_pos": 1, "dup_discrep": 1}
XGB_GRID = [{"max_depth": d, "n_estimators": n, "min_child_weight": m}
            for d in (2, 4) for n in (100, 300) for m in (5, 50)]
LR_GRID = [1e-3, 1e-2, 1e-1, 1.0]


# ---------------------------------------------------------------- L5 loop

def l5(rec, s):
    """Label-free loop L5: drop the worst |r|/unc line, re-solve the largest
    component; stop at max |r|/unc <= 3 or after ceil(0.05 n) removals.
    Its sequence is a prefix of the cached loop's (same deterministic rule),
    extended where the cache stops short. Returns order per comp line (-1 =
    never removed) and the removal count."""
    f = CACHE / (rec["name"].replace(":", "_").replace("@", "_") + "_L5.npz")
    if f.exists():
        z = np.load(f)
        return z["order"], int(z["c"])
    idx, order = s["idx"], s["order"]
    cap = int(np.ceil(0.05 * len(idx)))
    seq = [int(idx[j]) for j in np.argsort(order) if order[j] >= 0]
    k, k2 = s["k"], s["k2"]
    if k2 < 3 * k:                   # the cached loop reached the natural stop at k2
        c = min(k2, cap)
    elif cap <= len(seq):            # no natural stop within the cached removals
        c = cap
    else:                            # extend: replay the cached removals, continue
        tr = rec["trans"]
        cur = list(idx)
        for i in seq:
            cur = largest(tr, [j for j in cur if j != i])
        while len(seq) < cap:
            zz = resid(tr, cur) / np.array([tr[j].unc for j in cur])
            if zz.max() <= 3.0:
                break
            worst = cur[int(np.argmax(zz))]
            seq.append(worst)
            cur = largest(tr, [j for j in cur if j != worst])
        c = len(seq)
    pos = {i: n for n, i in enumerate(seq[:c])}
    out = np.array([pos.get(int(i), -1) for i in idx])
    np.savez(f, order=out, c=c)
    return out, c


# ---------------------------------------------------------------- features

YEAR = re.compile(r"^(\d\d)")


def year(tag):
    m = YEAR.match(tag)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 30 else 2000 + yy


def features(rec, s, l5_order, c5):
    """(n_comp, len(FEATURES)) array; label-free (see the pre-registration)."""
    tr = rec["trans"]
    idx = s["idx"]
    lines = [tr[i] for i in idx]
    n = len(lines)
    z = s["z"]
    unc = np.array([t.unc for t in lines])
    nu = np.array([t.freq for t in lines])
    F = {"log_z": np.log1p(z), "log_stud": np.log1p(s["stud"]), "h": s["h"],
         "loop_pos": np.where(l5_order >= 0, 1 - l5_order / max(c5, 1), 0.0)}
    deg = {}
    for t in lines:
        deg[t.upper] = deg.get(t.upper, 0) + 1
        deg[t.lower] = deg.get(t.lower, 0) + 1
    F["log_deg_u"] = np.log1p([deg[t.upper] for t in lines])
    F["log_deg_l"] = np.log1p([deg[t.lower] for t in lines])

    src = np.array([t.ref.rsplit(".", 1)[0] for t in lines])
    flagged = (z > 3) | (l5_order >= 0)
    for name in ("src_flagged", "src_scatter", "src_log_size", "src_share", "unc_rel_src"):
        F[name] = np.zeros(n)
    for sname in set(src):
        m = src == sname
        F["src_flagged"][m] = flagged[m].mean()
        F["src_scatter"][m] = np.log10(1.4826 * np.median(z[m]) + 1e-3)
        F["src_log_size"][m] = np.log10(m.sum())
        F["src_share"][m] = m.sum() / n
        F["unc_rel_src"][m] = np.log10(unc[m] / np.median(unc[m]))

    years = {sname: year(sname) for sname in set(src)}
    known = np.array(sorted(y for y in years.values() if y is not None))
    pct = {sname: (0.5 if y is None or len(known) < 2 else
                   ((known < y).sum() + 0.5 * ((known == y).sum() - 1)) / (len(known) - 1))
           for sname, y in years.items()}
    F["src_year_pct"] = np.array([pct[x] for x in src])
    F["year_missing"] = np.array([float(years[x] is None) for x in src])

    by_key = {}
    for k, t in enumerate(lines):
        by_key.setdefault((t.upper, t.lower), []).append(k)
    for name in ("has_dup", "n_dup_src", "best_other_ratio", "dup_discrep", "newer_dup"):
        F[name] = np.zeros(n)
    for ks in by_key.values():
        if len({src[k] for k in ks}) < 2:
            continue
        for k in ks:
            others = [j for j in ks if src[j] != src[k]]
            if not others:
                continue
            best = min(others, key=lambda j: unc[j])
            F["has_dup"][k] = 1.0
            F["n_dup_src"][k] = np.log1p(len({src[j] for j in others}))
            F["best_other_ratio"][k] = np.log10(unc[best] / unc[k])
            F["dup_discrep"][k] = np.log1p(abs(nu[k] - nu[best]) / np.hypot(unc[k], unc[best]))
            yk = years[src[k]]
            ys = [years[src[j]] for j in others]
            F["newer_dup"][k] = (np.mean([y > yk for y in ys if y is not None])
                                 if yk is not None and any(y is not None for y in ys) else 0.0)
    return np.column_stack([F[f] for f in FEATURES])


# ---------------------------------------------------------------- models

def train_rows(runs, groups):
    """Training matrix over the given groups: all positives + 20% of negatives
    per run (seed 0, weight x5); each group's total weight equal, split
    equally over its runs."""
    X, y, w = [], [], []
    by_group = {}
    for r in runs:
        if r["group"] in groups:
            by_group.setdefault(r["group"], []).append(r)
    total = sum(r["pos"].sum() + NEG_FRAC * (~r["pos"]).sum() for g in by_group for r in by_group[g])
    for g, rs in by_group.items():
        for r in rs:
            rng = np.random.default_rng(0)
            keep = r["pos"] | (rng.random(len(r["pos"])) < NEG_FRAC)
            wr = np.where(r["pos"][keep], 1.0, 1.0 / NEG_FRAC)
            wr *= total / len(by_group) / len(rs) / wr.sum()
            X.append(r["X"][keep])
            y.append(r["pos"][keep])
            w.append(wr)
    return np.vstack(X), np.concatenate(y).astype(float), np.concatenate(w)


def fit_xgb(X, y, w, hp, cols):
    """Native xgboost API (the sklearn wrapper would need scikit-learn)."""
    params = {"objective": "binary:logistic", "eta": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "lambda": 10, "tree_method": "hist", "seed": 0,
              "nthread": 8, "max_depth": hp["max_depth"], "min_child_weight": hp["min_child_weight"],
              "monotone_constraints": "(" + ",".join(str(MONOTONE.get(f, 0)) for f in cols) + ")"}
    return xgb.train(params, xgb.DMatrix(X, label=y, weight=w), num_boost_round=hp["n_estimators"])


def fit_lr(X, y, w, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    W = w / w.sum()

    def loss(b):
        t = Z @ b[1:] + b[0]
        p = expit(t)
        ll = W @ (np.logaddexp(0, t) - y * t)
        g = Z.T @ (W * (p - y))
        return ll + 0.5 * lam * b[1:] @ b[1:], np.r_[W @ (p - y), g + lam * b[1:]]

    b = minimize(loss, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B").x
    return lambda Xn: ((Xn - mu) / sd) @ b[1:] + b[0]


def predict(model, kind, X):
    return model.predict(xgb.DMatrix(X)) if kind == "xgb" else model(X)


def group_ap(runs, groups, preds):
    """Mean over groups of the mean per-run AP_ties (each group weighted 1)."""
    per = {}
    for r, p in zip(runs, preds):
        if r["group"] in groups:
            per.setdefault(r["group"], []).append(ap_ties(r["pos"], p))
    return float(np.mean([np.mean(v) for v in per.values()]))


def nested(runs, kind, cols):
    """Leave-one-group-out with nested LOGO tuning. Returns held-out score per
    run, the chosen setting and the fitted model per outer fold."""
    ci = [FEATURES.index(c) for c in cols]
    groups = sorted({r["group"] for r in runs})
    grid = XGB_GRID if kind == "xgb" else LR_GRID
    fit = (lambda X, y, w, hp: fit_xgb(X, y, w, hp, cols)) if kind == "xgb" else fit_lr
    held, chosen, models = [None] * len(runs), {}, {}
    for g_out in groups:
        t0 = time.time()
        train = [g for g in groups if g != g_out]
        score = []
        for hp in grid:
            inner = []
            for g_in in train:
                X, y, w = train_rows(runs, set(train) - {g_in})
                m = fit(X[:, ci], y, w, hp)
                pr = [predict(m, kind, r["X"][:, ci]) if r["group"] == g_in else None for r in runs]
                inner.append(group_ap(runs, {g_in}, pr))
            score.append(np.mean(inner))
        hp = grid[int(np.argmax(score))]
        X, y, w = train_rows(runs, set(train))
        m = fit(X[:, ci], y, w, hp)
        for k, r in enumerate(runs):
            if r["group"] == g_out:
                held[k] = predict(m, kind, r["X"][:, ci])
        chosen[g_out], models[g_out] = hp, m
        print(f"  {kind} {len(cols)}f fold {g_out}: {hp} inner AP {max(score):.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return held, chosen, models


# ---------------------------------------------------------------- metrics

def per_group(runs, values):
    """{group: mean over its runs} for per-run values (None skipped)."""
    by = {}
    for r, v in zip(runs, values):
        if v is not None and np.isfinite(v):
            by.setdefault(r["group"], []).append(v)
    return {g: float(np.mean(v)) for g, v in by.items()}


def pooled(by_group):
    groups = sorted(by_group)
    v = np.array([by_group[g] for g in groups])
    rng = np.random.default_rng(0)
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return {"mean": float(v.mean()), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
            "groups": len(groups), "by_group": {g: round(by_group[g], 4) for g in groups}}


def diff(a, b):
    """Paired molecule-weighted difference a - b over common groups, with CI."""
    return pooled({g: a[g] - b[g] for g in a if g in b})


def main():
    t_all = time.time()
    runs = []
    for rec in records():
        s = cached(rec)
        idx = s["idx"]
        neg, x = rec["neg"][idx], rec["x"][idx]
        n = neg & ~x
        if n.sum() < 10 or "outlier" not in rec["usable"]:
            continue
        o5, c5 = l5(rec, s)
        X = features(rec, s, o5, c5)
        pool = ~x
        order = s["order"]
        k, k2 = s["k"], s["k2"]
        caught2 = (order >= 0) & (order < k2)
        bridge = s["h"] >= 0.99
        og = np.array([rec["trans"][i].unc for i in idx])
        used = rec["used"][idx]
        two_col = rec["u_label"] is None and "uncertainty" in rec["usable"]
        runs.append({
            "name": rec["name"], "group": rec["group"], "X": X[pool], "pos": n[pool],
            "z": s["z"][pool], "K": int(n.sum()),
            "base": {"iv2": np.where(caught2, -order, -len(idx))[pool].astype(float),
                     "iv": np.where((order >= 0) & (order < k), -order, -len(idx))[pool].astype(float),
                     "L5": np.where(o5 >= 0, -o5, -len(idx))[pool].astype(float),
                     "i_resid": s["z"][pool]},
            "H": (n & ~bridge & ~caught2 & ~s["rule"])[pool],
            "H_iv": (n & ~bridge & ~((order >= 0) & (order < k)) & ~s["rule"])[pool],
            "raised": ((used > og * (1 + 1e-9)) & ~neg)[pool] if two_col else None,
            "c5": c5})
        print(f"{rec['name']}: {pool.sum()} lines, N={n.sum()}, L5 removals {c5}", flush=True)

    res = {"runs": [r["name"] for r in runs], "models": {}}
    held = {}
    for name, kind, cols in (("XGB", "xgb", FEATURES), ("LR", "lr", FEATURES),
                             ("XGB_nodup", "xgb", [f for f in FEATURES if f not in FAMILY["dup"]])):
        h, chosen, models = nested(runs, kind, cols)
        held[name] = (h, models, cols)
        res["models"][name] = {"chosen": {g: (hp if kind == "xgb" else {"lambda": hp})
                                          for g, hp in chosen.items()}}

    scores = {nm: v[0] for nm, v in held.items()}
    scores |= {b: [r["base"][b] for r in runs] for b in ("iv2", "iv", "L5", "i_resid")}
    ap = {nm: per_group(runs, [ap_ties(r["pos"], p) for r, p in zip(runs, sc)])
          for nm, sc in scores.items()}
    res["AP_per_run"] = {nm: {r["name"]: ap_ties(r["pos"], p) for r, p in zip(runs, sc)}
                         for nm, sc in scores.items()}
    res["AP_pooled"] = {nm: pooled(v) for nm, v in ap.items()}
    res["primary"] = {"XGB-iv2": diff(ap["XGB"], ap["iv2"]), "LR-iv2": diff(ap["LR"], ap["iv2"])}
    res["secondary_L5"] = {"XGB-L5": diff(ap["XGB"], ap["L5"]), "LR-L5": diff(ap["LR"], ap["L5"])}
    res["dup_contribution"] = {"XGB-XGB_nodup": diff(ap["XGB"], ap["XGB_nodup"])}

    # headroom recovery: share of H lines in the top K = |N| (tie-aware)
    rec_h = {}
    for hname in ("H", "H_iv"):
        for split in ("all", "visible", "invisible"):
            for nm, sc in scores.items():
                vals = []
                for r, p in zip(runs, sc):
                    m = r[hname] & ({"all": True, "visible": r["z"] >= 3,
                                     "invisible": r["z"] < 3}[split])
                    vals.append(recall_ties(m, p, r["K"]) if m.sum() else None)
                rec_h[f"{hname}.{split}.{nm}"] = pooled(per_group(runs, vals)) if any(
                    v is not None for v in vals) else None
            vals = [(r["K"] / len(r["pos"]) if (r[hname] & ({"all": True, "visible": r["z"] >= 3,
                                                             "invisible": r["z"] < 3}[split])).sum()
                     else None) for r in runs]
            rec_h[f"{hname}.{split}.random"] = pooled(per_group(runs, vals))
    res["H_recovery"] = rec_h

    # negate vs inflate among |r|/unc >= 3 lines in 2-column runs
    nvi = {}
    for nm in ("XGB", "LR", "i_resid", "L5"):
        vals, counts = [], {}
        for r, p in zip(runs, scores[nm]):
            if r["raised"] is None:
                vals.append(None)
                continue
            m = (r["z"] >= 3) & (r["pos"] | r["raised"])
            lab = r["pos"][m]
            counts[r["name"]] = [int(lab.sum()), int((~lab).sum())]
            vals.append(auroc(lab, p[m]) if 0 < lab.sum() < m.sum() else None)
        nvi[nm] = pooled(per_group(runs, vals))
        nvi["counts"] = counts
    res["negate_vs_inflate"] = nvi

    # importance: XGB gain + permutation on held-out folds (pooled AP drop)
    gain = {}
    for g, m in held["XGB"][1].items():
        for f, v in m.get_score(importance_type="total_gain").items():
            gain.setdefault(FEATURES[int(f[1:])], []).append(v)
    tot = sum(np.mean(v) for v in gain.values())
    res["xgb_gain_share"] = {f: float(np.mean(gain.get(f, [0])) / tot) for f in FEATURES}
    base_ap = res["AP_pooled"]["XGB"]["mean"]
    perm = {}
    for label, cols in [(f, [f]) for f in FEATURES] + [(f"family:{k}", v) for k, v in FAMILY.items()]:
        ci = [FEATURES.index(c) for c in cols]
        drops = []
        for rep in range(3):
            rng = np.random.default_rng(rep)
            preds = []
            for r in runs:
                Xp = r["X"].copy()
                Xp[:, ci] = Xp[rng.permutation(len(Xp))][:, ci]
                preds.append(held["XGB"][1][r["group"]].predict(xgb.DMatrix(Xp)))
            drops.append(base_ap - np.mean(list(per_group(
                runs, [ap_ties(r["pos"], p) for r, p in zip(runs, preds)]).values())))
        perm[label] = float(np.mean(drops))
    res["permutation_AP_drop"] = perm

    # decision rule (pre-registered; judged on iv')
    a = res["primary"]["XGB-iv2"]
    b = rec_h["H.all.XGB"]["mean"]
    b_rand = rec_h["H.all.random"]["mean"]
    dup_gain = res["dup_contribution"]["XGB-XGB_nodup"]["mean"]
    cond_a = a["mean"] > 0 and a["ci95"][0] > 0
    cond_b = b >= 0.25 and b >= 2 * b_rand
    res["decision"] = {
        "a_XGB_beats_iv2": cond_a, "b_H_recovery": b, "b_random": b_rand, "b_ok": cond_b,
        "dup_share_of_gain": dup_gain / a["mean"] if a["mean"] > 0 else None,
        "verdict": ("escalate (tabular hybrid: dup family carries the gain)"
                    if cond_a and cond_b and a["mean"] > 0 and dup_gain >= 0.5 * a["mean"] else
                    "escalate" if cond_a and cond_b else
                    "no escalation; possible tabular re-ranker" if cond_a else
                    "no escalation; the loop stays the tool")}
    res["runtime_s"] = time.time() - t_all
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    report(res)


def report(res):
    L = ["AP (tie-aware), pooled molecule-weighted, 95% cluster bootstrap over groups"]
    for nm, v in res["AP_pooled"].items():
        L.append(f"  {nm:10s} {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]  {v['by_group']}")
    for sec in ("primary", "secondary_L5", "dup_contribution"):
        L.append(sec)
        for nm, v in res[sec].items():
            L.append(f"  {nm:16s} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['by_group']}")
    L.append("H recovery (share of headroom lines in top K = |N|)")
    for k, v in res["H_recovery"].items():
        if v:
            L.append(f"  {k:28s} {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] ({v['groups']} groups)")
    L.append("negate vs inflate AUROC (|r|/unc >= 3, 2-column runs)")
    for k, v in res["negate_vs_inflate"].items():
        L.append(f"  {k:8s} {json.dumps(v)}")
    L.append("XGB gain share: " + json.dumps({k: round(v, 3) for k, v in res["xgb_gain_share"].items()}))
    L.append("permutation AP drop: " + json.dumps({k: round(v, 4) for k, v in res["permutation_AP_drop"].items()}))
    L.append("chosen hyperparameters: " + json.dumps({k: v["chosen"] for k, v in res["models"].items()}))
    L.append("decision: " + json.dumps(res["decision"], default=float))
    text = "\n".join(L)
    (OUT / "results.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
