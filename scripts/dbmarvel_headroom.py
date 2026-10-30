"""db_MARVEL headroom study, phase 2: how much of the curators' outlier and
uncertainty work does the no-model pipeline (iterative loop + generic rules)
already recover?

Run: uv run python scripts/dbmarvel_headroom.py

Pre-registered in reports/2026-10-30_dbmarvel_headroom/PREREGISTRATION.md.
Per-run line scores are cached in data/dbmarvel/scores/ (delete to recompute);
results go to reports/2026-10-30_dbmarvel_headroom/{results.json,results.txt}.
"""

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from marvel_gnn.core.parse import UNC_FLOOR, Transition
from marvel_gnn.core.solver import level_index, solve_energies
from marvel_gnn.gnn.data import edge_leverages

sys.path.insert(0, str(Path(__file__).parent))
from c2h4_blind_score import ap, auroc, deliverable, load_key, n_categories, recall_at  # noqa: E402
from dbmarvel_audit import DB, OUT, ROOT, load, runs  # noqa: E402

CACHE = DB / "scores"
MIN_POS = 10
GROUP = {"CO2": "CO2", "CO": "CO", "VO": "VO", "H3+": "H3+", "H2D+": "H3+", "D2H+": "H3+"}
N_BOOT = 1000


# ---------------------------------------------------------------- run records

def usable_rows():
    """audit.tsv 'usable' column, in runs() order (C2H4 last)."""
    rows = (OUT / "audit.tsv").read_text().splitlines()
    head = rows[0].split("\t")
    return [dict(zip(head, r.split("\t")))["usable"] for r in rows[1:]]


def og_copy(trans):
    return [Transition(t.freq, t.orig_unc, t.orig_unc, t.upper, t.lower, t.ref) for t in trans]


def records():
    """One dict per usable run: trans (unc = original), neg (N or X), X,
    used unc, U-pool exclusions, qn names, group."""
    use = usable_rows()
    for (pub, d, units), u in zip(runs(), use):
        if u == "no":
            continue
        trans, neg = load(pub, d, units)
        tags = np.array([t.ref.rsplit(".", 1)[0] for t in trans])
        whole = {s for s in set(tags[neg]) if neg[tags == s].all()}
        x = np.array([s in whole for s in tags])
        used = np.array([t.unc for t in trans])
        yield {"name": f"{pub}:{d['isotopologue']}:{len(d['qn_names'])}", "formula": d["formula"],
               "group": GROUP.get(d["formula"], d["formula"]), "usable": u,
               "trans": og_copy(trans), "neg": neg, "x": x, "used": used,
               "u_label": None, "u_exclude": np.zeros(len(trans), bool),
               "qn": d["qn_names"], "cat": None}
    yield c2h4_record()


def c2h4_record():
    """12C2H4 v1: unedited lines as in the blind test; N = key negations +
    source negations outside 58AlPl; X = 58AlPl; U = line-level set_unc."""
    from c2h4_blind_predict import load_unedited
    lines, flipped = load_unedited()
    key = load_key(ROOT / "data/C2H4_v1/key/overrides_0b56238.tsv")
    deliv = deliverable()
    refs = [t.ref for t in lines]
    x = np.array([r.startswith("58AlPl.") for r in refs])
    neg = np.array([r in key["negate"] or r in flipped for r in refs]) | x
    cat = ["/".join(sorted(n_categories(key["negate"][r][1]))) if r in key["negate"]
           else ("S" if r in flipped else "") for r in refs]
    used = np.array([deliv[r][2] if r in deliv else t.unc for r, t in zip(refs, lines)])
    u_label = np.array([r in key["set_unc"] for r in refs])
    scaled = np.array([r.rsplit(".", 1)[0] in key["scale"] for r in refs])
    return {"name": "MARVEL_network@0b56238:12C2H4:15", "formula": "C2H4", "group": "C2H4",
            "usable": "outlier+uncertainty", "trans": og_copy(lines), "neg": neg, "x": x,
            "used": used, "u_label": u_label, "u_exclude": scaled & ~u_label,
            "qn": [f"v{i}" for i in range(1, 13)] + ["J", "Ka", "Kc"], "cat": np.array(cat)}


# ---------------------------------------------------------------- scoring

def largest(trans, keep):
    """Indices (from `keep`) of the lines in the largest component of trans[keep]."""
    g = nx.Graph((trans[i].upper, trans[i].lower) for i in keep)
    main = max(nx.connected_components(g), key=len)
    return [i for i in keep if trans[i].upper in main]


def resid(trans, keep, unc=None):
    sub = [trans[i] for i in keep]
    e = solve_energies(sub, unc)
    return np.array([abs(t.freq - (e[t.upper] - e[t.lower])) for t in sub])


def loop(trans, comp, k):
    """Drop the worst |r|/unc line and re-solve the largest component, until
    both (iv) has its k removals and (iv') has stopped (max |r|/unc <= 3, or
    3k removals). Returns (removal order, k', lines left after k removals)."""
    cur, order, stop2, pruned = list(comp), [], None, list(comp) if k == 0 else None
    while True:
        z = resid(trans, cur) / np.array([trans[i].unc for i in cur])
        if stop2 is None and (z.max() <= 3.0 or len(order) >= 3 * k):
            stop2 = len(order)
        if len(order) >= k and stop2 is not None:
            return order, stop2, pruned
        worst = cur[int(np.argmax(z))]
        order.append(worst)
        cur = largest(trans, [i for i in cur if i != worst])
        if len(order) == k:
            pruned = list(cur)


def replay(trans, keep, rounds=100):
    """MARVEL 4-style optimisation: raise unc to |r| where |r| > unc, re-solve,
    until no unc moves by > 0.1%. Returns final unc / original, per kept line."""
    og = np.array([trans[i].unc for i in keep])
    unc = og.copy()
    for _ in range(rounds):
        r = resid(trans, keep, list(unc))
        new = np.maximum(unc, r)
        if (new / unc - 1).max() <= 1e-3:
            unc = new
            break
        unc = new
    return unc / og


def _num(tok):
    try:
        return float(tok)
    except ValueError:
        return None


def rules(trans, comp, names):
    """Pre-registered generic rules R1-R5 by QN schema. Returns (fired, applicable)
    bool arrays over comp. Unreadable tokens skip that rule for that line."""
    ix = {n: i for i, n in enumerate(names)}
    fired = np.zeros(len(comp), bool)
    applic = np.zeros(len(comp), bool)
    for k, i in enumerate(comp):
        u, l = trans[i].upper.split(), trans[i].lower.split()
        if len(u) != len(names) or len(l) != len(names) or "J" not in ix:
            continue
        ju, jl = _num(u[ix["J"]]), _num(l[ix["J"]])
        if ju is None or jl is None:
            continue
        applic[k] = True
        dj = abs(ju - jl)
        f = dj > 1 + 1e-9                                                  # R1
        if "ef" in ix and {u[ix["ef"]], l[ix["ef"]]} <= {"e", "f"}:         # R2
            same = u[ix["ef"]] == l[ix["ef"]]
            f |= (dj < 1e-9 and same) or (abs(dj - 1) < 1e-9 and not same)
        if "parity" in ix and {u[ix["parity"]], l[ix["parity"]]} <= {"+", "-"}:  # R2'
            f |= u[ix["parity"]] == l[ix["parity"]]
        if "l2" in ix:                                                     # R3
            for lev, j in ((u, ju), (l, jl)):
                l2 = _num(lev[ix["l2"]])
                if l2 is not None:
                    f |= j < l2 or (l2 == 0 and "ef" in ix and lev[ix["ef"]] == "f")
        if "Ka" in ix and "Kc" in ix:                                      # R4, R5
            ka = [_num(u[ix["Ka"]]), _num(l[ix["Ka"]])]
            kc = [_num(u[ix["Kc"]]), _num(l[ix["Kc"]])]
            if None not in ka + kc:
                for a, c, j in ((ka[0], kc[0], ju), (ka[1], kc[1], jl)):
                    f |= a + c not in (j, j + 1) or a > j or c > j
                f |= (ka[0] - ka[1]) % 2 == 0 and (kc[0] - kc[1]) % 2 == 0
        fired[k] = f
    return fired, applic


def score_run(rec):
    """Per-line scores over the largest component of the unedited graph."""
    tr, neg = rec["trans"], rec["neg"]
    comp = largest(tr, range(len(tr)))
    r = resid(tr, comp)
    og = np.array([tr[i].unc for i in comp])
    out = {"idx": np.array(comp), "z": r / og}
    out["rule"], out["rule_applicable"] = rules(tr, comp, rec["qn"])
    k = int(neg[comp].sum())
    t0 = time.time()
    order, stop2, pruned = loop(tr, comp, k)
    out["loop_s"] = time.time() - t0
    pos = {i: n for n, i in enumerate(order)}
    out["order"] = np.array([pos.get(i, -1) for i in comp])
    out["k"], out["k2"] = k, stop2
    if "uncertainty" in rec["usable"]:
        tags = np.array([tr[i].ref.rsplit(".", 1)[0] for i in comp])
        s = {t: 1.4826 * np.median(out["z"][tags == t]) for t in set(tags)}
        out["s_src"] = np.array([s[t] for t in tags])
        removed = set(order[:k])
        p = dict(zip(pruned, replay(tr, pruned)))
        out["P"] = np.array([np.inf if i in removed else p.get(i, 1.0) for i in comp])
        edited = largest(tr, [i for i in comp if not neg[i]])
        ps = dict(zip(edited, replay(tr, edited)))
        out["Pstar"] = np.array([ps.get(i, 1.0) for i in comp])
    return out


def leverage(rec, s):
    """h and studentized residual, recomputed on every load (cheap, and not
    cached: the loop and replay don't use them). Leverage uses unc floored at
    UNC_FLOOR: PN's 4.6e-10..0.2 cm-1 (weights over ~1e17) breaks the factor's
    precision (sum h = 1568 vs 1661). Deviation from the pre-registration,
    applied to every run; residuals stay unfloored."""
    tr = rec["trans"]
    sub = [Transition(tr[i].freq, max(tr[i].unc, UNC_FLOOR), 0.0, tr[i].upper, tr[i].lower, "")
           for i in s["idx"]]
    s["h"] = edge_leverages(sub, level_index(sub))
    s["stud"] = s["z"] / np.sqrt(np.maximum(1 - s["h"], 1e-12))
    return s


def cached(rec):
    return leverage(rec, _cached(rec))


def _cached(rec):
    f = CACHE / (rec["name"].replace(":", "_").replace("@", "_") + ".npz")
    if f.exists():
        z = np.load(f)
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    t0 = time.time()
    out = score_run(rec)
    CACHE.mkdir(exist_ok=True)
    np.savez(f, **out)
    print(f"  scored {rec['name']} in {time.time() - t0:.0f} s (loop {out['loop_s']:.0f} s, "
          f"k={out['k']}, k'={out['k2']})", flush=True)
    return out


# ---------------------------------------------------------------- metrics

def ap_ties(labels, scores):
    """AP as a step function over distinct score thresholds: a tied block
    counts as one step (expected precision under a random order within it).
    The pre-registered line-order tie-break rewards files whose negated lines
    happen to sit early (rules: AP 1.000 with 0 hits on positives)."""
    o = np.argsort(-scores, kind="stable")
    sc, lab = scores[o], labels[o]
    last = np.r_[np.nonzero(np.diff(sc))[0], len(sc) - 1]  # end of each tie block
    tp = np.cumsum(lab)[last]
    prec = tp / (last + 1)
    return float((np.diff(np.r_[0, tp]) * prec).sum() / lab.sum())


def recall_ties(labels, scores, n):
    """Expected recall in the top n when the boundary tie block is ordered at random."""
    srt = np.sort(scores)[::-1]
    cut = srt[n - 1]
    above = scores > cut
    tied = scores == cut
    share = (n - above.sum()) / tied.sum()
    return float((labels[above].sum() + share * labels[tied].sum()) / labels.sum())


def outlier_metrics(rec, s):
    idx = s["idx"]
    neg, x = rec["neg"][idx], rec["x"][idx]
    n = neg & ~x
    out = {"N": int(n.sum()), "X": int(x.sum()), "lines": len(idx)}
    if n.sum() < MIN_POS:
        return out
    order = s["order"]
    k, k2 = s["k"], s["k2"]
    caught = (order >= 0) & (order < k)
    caught2 = (order >= 0) & (order < k2)
    pool = ~x
    lab = n[pool]
    scores = {"i_resid": s["z"], "ii_stud": s["stud"],
              "iv_loop": np.where(caught, -order, -len(idx)).astype(float),
              "iv2_loop": np.where(caught2, -order, -len(idx)).astype(float),
              "R_rules": s["rule"].astype(float)}
    top = int(np.ceil(0.01 * pool.sum()))
    out["scores"] = {nm: {"AP": ap(lab, v[pool]), "recall@K": recall_at(lab, v[pool], int(lab.sum())),
                          "recall_top1%": recall_at(lab, v[pool], top),
                          "AP_ties": ap_ties(lab, v[pool]),
                          "recall@K_ties": recall_ties(lab, v[pool], int(lab.sum())),
                          "recall_top1%_ties": recall_ties(lab, v[pool], top)}
                     for nm, v in scores.items()}
    out["prevalence"] = float(lab.mean())
    out["loop_precision_K"] = float(neg[caught].mean()) if k else None
    out["iv2"] = {"removals": int(k2), "precision": float(neg[caught2].mean()) if k2 else None,
                  "recall_NX": float(caught2[neg].mean())}
    bridge = s["h"] >= 0.99
    for tag, c in (("iv", caught), ("iv2", caught2)):
        part = {}
        for nm, m in (("N", n), ("X", x)):
            if not m.any():
                continue
            b = m & bridge
            lp = m & ~bridge & c
            rr = m & ~bridge & ~c & s["rule"]
            hh = m & ~bridge & ~c & ~s["rule"]
            part[nm] = {"B": int(b.sum()), "L": int(lp.sum()), "R": int(rr.sum()), "H": int(hh.sum()),
                        "B&rule": int((b & s["rule"]).sum()), "L&rule": int((lp & s["rule"]).sum())}
        out[f"partition_{tag}"] = part
    out["rule_applicable"] = float(s["rule_applicable"].mean())
    out["rule_fired_negatives"] = int((s["rule"] & ~neg).sum())
    if rec["cat"] is not None:  # C2H4 key categories
        cats = rec["cat"][idx]
        by = {}
        for c in ("QAB", "QAI/QAM", "QAC", "QAJ", "QAU", "S"):
            m = n & np.array([c in v.split("/") if c != "QAI/QAM" else "QAI/QAM" in v for v in cats])
            if c == "S":
                m = n & (cats == "S")
            by[c] = {"n": int(m.sum()), "B": int((m & bridge).sum()),
                     "L_iv": int((m & ~bridge & caught).sum()), "L_iv2": int((m & ~bridge & caught2).sum()),
                     "R": int((m & ~bridge & s["rule"]).sum())}
        out["c2h4_categories"] = by
    return out


def unc_metrics(rec, s):
    if "P" not in s:
        return None
    idx = s["idx"]
    og = np.array([rec["trans"][i].unc for i in idx])
    used = rec["used"][idx]
    lowered = used < og * (1 - 1e-9)
    if rec["u_label"] is None:
        lab_all = used > og * (1 + 1e-9)
    else:
        lab_all = rec["u_label"][idx]
    pool = ~rec["neg"][idx] & ~lowered & ~rec["u_exclude"][idx]
    lab = lab_all[pool]
    if lab.sum() < MIN_POS:
        return None
    factor = (used / og)[pool]
    scores = {"inv_unc": 1 / og, "i_resid": s["z"], "ii_stud": s["stud"], "s_src": s["s_src"],
              "P": s["P"], "Pstar": s["Pstar"]}
    out = {"U": int(lab.sum()), "pool": int(pool.sum()), "lowered": int(lowered.sum()), "scores": {}}
    for nm, v in scores.items():
        v = v[pool]
        rho = spearmanr(factor[lab], v[lab]).statistic if np.ptp(v[lab]) > 0 else float("nan")
        out["scores"][nm] = {"AUROC": auroc(lab, np.nan_to_num(v, posinf=1e300)), "rho": float(rho)}
    pr = s["P"][pool] > 1.001
    out["P_raised"] = {"n": int(pr.sum()), "precision": float(lab[pr].mean()) if pr.any() else None,
                       "recall": float(pr[lab].mean())}
    return out


# ---------------------------------------------------------------- pooling

def pooled(per_run, value, rng_seed=0):
    """Molecule-weighted mean of value(run) over runs where it is defined, with
    a 95% cluster bootstrap over groups. per_run: list of (group, result)."""
    vals = [(g, value(r)) for g, r in per_run]
    vals = [(g, v) for g, v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    groups = sorted({g for g, _ in vals})
    by = {g: [v for gg, v in vals if gg == g] for g in groups}
    gmean = np.array([np.mean(by[g]) for g in groups])  # weight 1 per group, split over its runs
    rng = np.random.default_rng(rng_seed)
    boots = [gmean[rng.integers(0, len(groups), len(groups))].mean() for _ in range(N_BOOT)]
    return {"mean": float(gmean.mean()), "ci95": [float(v) for v in np.percentile(boots, [2.5, 97.5])],
            "groups": len(groups), "by_group": dict(zip(groups, map(float, gmean)))}


def frac(part, b):
    tot = sum(part[k] for k in "BLRH")
    return part[b] / tot if tot else None


def main():
    recs, outl, unc = [], [], []
    for rec in records():
        print(rec["name"], flush=True)
        s = cached(rec)
        o, u = outlier_metrics(rec, s), unc_metrics(rec, s)
        recs.append({"run": rec["name"], "group": rec["group"], "outlier": o, "uncertainty": u})
        if "scores" in o and "outlier" in rec["usable"]:
            outl.append((rec["group"], o))
        if u:
            unc.append((rec["group"], u))

    res = {"runs": recs, "pooled": {}}
    P = res["pooled"]
    for tag in ("iv", "iv2"):
        P[f"partition_{tag}"] = {b: pooled(outl, lambda o, b=b, t=tag: frac(o[f"partition_{t}"]["N"], b))
                                 for b in "BLRH"}
        tot = {b: sum(o[f"partition_{tag}"]["N"][b] for _, o in outl) for b in "BLRH"}
        P[f"partition_{tag}_unweighted_lines"] = {b: tot[b] / sum(tot.values()) for b in "BLRH"}
    for nm in ("i_resid", "ii_stud", "iv_loop", "iv2_loop", "R_rules"):
        P[f"AP_{nm}"] = pooled(outl, lambda o, nm=nm: o["scores"][nm]["AP"])
        P[f"AP_ties_{nm}"] = pooled(outl, lambda o, nm=nm: o["scores"][nm]["AP_ties"])
        P[f"recall@K_ties_{nm}"] = pooled(outl, lambda o, nm=nm: o["scores"][nm]["recall@K_ties"])
    P["AP_random"] = pooled(outl, lambda o: o["prevalence"])
    for nm in ("inv_unc", "i_resid", "ii_stud", "s_src", "P", "Pstar"):
        P[f"AUROC_{nm}"] = pooled(unc, lambda u, nm=nm: u["scores"][nm]["AUROC"])
        P[f"rho_{nm}"] = pooled(unc, lambda u, nm=nm: u["scores"][nm]["rho"])

    verdict = {}
    for tag in ("iv", "iv2"):
        h = P[f"partition_{tag}"]["H"]
        big = sum(v >= 0.25 for v in h["by_group"].values())
        verdict[tag] = ("not worth running" if h["mean"] < 0.20 else
                        "worth running" if h["mean"] >= 0.40 and big >= 4 else "judgement call")
        verdict[f"{tag}_groups_H>=0.25"] = big
    P["verdict"] = verdict
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    report(res)


def report(res):
    L = []
    for r in res["runs"]:
        o, u = r["outlier"], r["uncertainty"]
        line = f"{r['run']:42s} N={o['N']:5d} X={o['X']:5d}"
        if "scores" in o:
            sc = o["scores"]
            pa, pb = o["partition_iv"]["N"], o["partition_iv2"]["N"]
            line += (" | AP_ties " + " ".join(f"{k.split('_')[0]} {v['AP_ties']:.3f}" for k, v in sc.items()))
            line += (f" | AP i {sc['i_resid']['AP']:.3f} ii {sc['ii_stud']['AP']:.3f} iv "
                     f"{sc['iv_loop']['AP']:.3f} iv' {sc['iv2_loop']['AP']:.3f} R {sc['R_rules']['AP']:.3f}"
                     f" rand {o['prevalence']:.3f} | iv B{pa['B']} L{pa['L']} R{pa['R']} H{pa['H']}"
                     f" | iv' k'={o['iv2']['removals']} L{pb['L']} R{pb['R']} H{pb['H']}"
                     f" | rule-applic {o['rule_applicable']:.2f} fired-neg {o['rule_fired_negatives']}")
        if u:
            sc = u["scores"]
            line += (f"\n{'':42s} U={u['U']} pool={u['pool']} AUROC 1/unc {sc['inv_unc']['AUROC']:.3f}"
                     f" i {sc['i_resid']['AUROC']:.3f} ii {sc['ii_stud']['AUROC']:.3f} s_src "
                     f"{sc['s_src']['AUROC']:.3f} P {sc['P']['AUROC']:.3f} P* {sc['Pstar']['AUROC']:.3f}"
                     f" | rho P {sc['P']['rho']:.2f} P* {sc['Pstar']['rho']:.2f} s_src {sc['s_src']['rho']:.2f}"
                     f" | P raised n={u['P_raised']['n']} prec {u['P_raised']['precision']} "
                     f"rec {u['P_raised']['recall']:.3f}")
        L.append(line)
    L.append("\nPOOLED (molecule-weighted; 95% cluster bootstrap over groups)")
    for k, v in res["pooled"].items():
        if isinstance(v, dict) and "mean" in v:
            L.append(f"{k:28s} {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] ({v['groups']} groups)")
        elif k.startswith("partition") and "unweighted" not in k:
            for b, w in v.items():
                L.append(f"{k}.{b:2s}{'':10s} {w['mean']:.3f} [{w['ci95'][0]:.3f}, {w['ci95'][1]:.3f}]"
                         f" by group {json.dumps({g: round(x, 3) for g, x in w['by_group'].items()})}")
        else:
            L.append(f"{k:28s} {json.dumps(v, default=float)}")
    text = "\n".join(L)
    (OUT / "results.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
