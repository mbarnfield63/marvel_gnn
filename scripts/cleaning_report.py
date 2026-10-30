"""Step 2 aggregation: per-run metrics, molecule-weighted pooling, dominance,
Pareto front, MILP gaps (see cleaning_optimality.py for the stages).

Run: uv run python scripts/cleaning_optimality.py report
"""

import json

import numpy as np

from cleaning_optimality import (BASE, CACHE, KFREE, KS, OUT, PHASE_B, REMOVAL, load_out, load_run,
                                 run_names, safe)
from dbmarvel_headroom import ap_ties

METHODS = KFREE + BASE + PHASE_B
N_BOOT = 1000
C2H4_SCALE = {"02MoKoBuHe": 3, "83DaPiFaDe": 5, "19UlBeGrRa": 2, "24BeVaTuBo": 2, "25BeBeReCa": 2}


def load_score(rname, name, k):
    kk = "x" if name in KFREE else f"{k:g}"
    p = CACHE / f"score_{rname}_{name}_k{kk}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {f: (z[f].item() if z[f].ndim == 0 else z[f]) for f in z.files}


def run_metrics(run, name, k):
    rname = safe(run["name"])
    try:
        o = load_out(run, name, k)
    except FileNotFoundError:
        return None
    s = load_score(rname, name, k)
    keep, u, u0, neg = o["keep"], o["u"], run["u0"], run["neg"]
    m = {"kept": float(keep.mean()), "secs": o["secs"],
         "own": int((o["why"] == 1).sum()), "finish": int((o["why"] == 2).sum()),
         "cut": int((o["why"] == 3).sum()), "readmitted": o["extra"].get("readmitted", 0),
         "swamp": o["extra"].get("swamp", 0)}
    infl = keep & (u > u0 * (1 + 1e-9))
    m["inflated"] = int(infl.sum())
    m["infl_median"] = float(np.median(u[infl] / u0[infl])) if infl.any() else 1.0
    removed = ~keep
    m["cur_precision"] = float(neg[removed].mean()) if removed.any() else None
    m["cur_recall"] = float(removed[neg].mean())
    m["cur_ap"] = ap_ties(neg, np.where(keep, u / u0, np.inf))
    if s is not None:
        sc = ~np.isnan(s["nll"])
        m |= {"nll": float(np.nanmean(s["nll"])), "cov1": float((np.abs(s["z"][sc]) <= 1).mean()),
              "cov2": float((np.abs(s["z"][sc]) <= 2).mean()), "disc": int(s["disc"]),
              "validated": s["validated"] / run["n"], "orphaned": int(run["n"] - s["levels"]),
              "sigma_med": s["sigma_med"], "maxz": s["maxz"], "score_secs": s["secs"]}
        src = run["src"]
        good, tot = 0, 0
        for x in np.unique(src[sc]):
            mm = sc & (src == x)
            if mm.sum() >= 10:
                tot += 1
                good += 0.5 <= np.mean(s["z"][mm] ** 2) <= 2
        m["src_calibrated"] = good / tot if tot else None
        m["_nll_line"] = s["nll"]
    return m


def groups_of(runs):
    return {r["name"]: str(r["group"]) for r in runs}


def pooled(values, grp):
    """values: {run name: value}; molecule-weighted mean with a cluster bootstrap."""
    by = {}
    for r, v in values.items():
        if v is not None and np.isfinite(v):
            by.setdefault(grp[r], []).append(v)
    if not by:
        return None
    g = sorted(by)
    vals = np.array([np.mean(by[x]) for x in g])
    rng = np.random.default_rng(0)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(N_BOOT)]
    return {"mean": float(vals.mean()), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
            "groups": len(g), "by_group": {x: round(float(v), 4) for x, v in zip(g, vals)}}


def dominates(d):
    """d: pooled differences X - Y for nll, kept, validated. Pre-registered rule."""
    if not all(d.get(a) for a in ("nll", "kept", "validated")):
        return False
    no_worse = (d["nll"]["ci95"][1] < 0.05 and d["kept"]["mean"] >= -0.002
                and d["validated"]["mean"] >= -0.002)
    strict = (d["nll"]["ci95"][1] < 0 or d["kept"]["ci95"][0] > 0 or d["validated"]["ci95"][0] > 0)
    return bool(no_worse and strict)


def report():
    runs = [load_run(n) for n in run_names()]
    grp = groups_of(runs)
    res = {"k": {}}
    for k in KS:
        per = {name: {} for name in METHODS}
        for run in runs:
            for name in METHODS:
                per[name][run["name"]] = run_metrics(run, name, k)
        # common set: lines kept and scored by every method with a score
        common = {}
        for run in runs:
            ms = [per[n][run["name"]] for n in METHODS if per[n][run["name"]] and "nll" in per[n][run["name"]]]
            if not ms:
                continue
            c = np.ones(len(run["nu"]), bool)
            for mm in ms:
                c &= ~np.isnan(mm["_nll_line"])
            common[run["name"]] = c
            for n in METHODS:
                mm = per[n][run["name"]]
                if mm and "nll" in mm:
                    mm["nll_common"] = float(np.mean(mm["_nll_line"][c])) if c.any() else None
                    mm["n_common"] = int(c.sum())
        for n in METHODS:
            for mm in per[n].values():
                if mm:
                    mm.pop("_nll_line", None)

        def pool(metric, name):
            return pooled({r: (v or {}).get(metric) for r, v in per[name].items()}, grp)

        def diff(metric, a, b):
            return pooled({r: (per[a][r][metric] - per[b][r][metric]
                               if per[a][r] and per[b][r] and per[a][r].get(metric) is not None
                               and per[b][r].get(metric) is not None else None) for r in per[a]}, grp)

        kres = {"methods": {}, "vs_M1": {}, "vs_M0": {}, "per_run": per}
        metrics = ["nll", "nll_common", "kept", "validated", "orphaned", "inflated", "infl_median", "cov1",
                   "cov2", "src_calibrated", "disc", "sigma_med", "maxz", "own", "finish", "cut",
                   "readmitted", "swamp", "cur_precision", "cur_recall", "cur_ap", "secs"]
        for n in METHODS:
            kres["methods"][n] = {mt: pool(mt, n) for mt in metrics}
            for base in ("M1", "M0"):
                kres[f"vs_{base}"][n] = {mt: diff(mt, n, base) for mt in ("nll", "nll_common", "kept",
                                                                           "validated")}
        # dominance and front
        dom = {}
        for a in METHODS:
            for b in METHODS:
                if a != b:
                    dd = {mt: diff(mt, a, b) for mt in ("nll", "kept", "validated")}
                    if dominates(dd):
                        dom.setdefault(a, []).append(b)
        front = [n for n in METHODS if not any(n in v for v in dom.values())]
        beats_m1 = [n for n in front if "M1" in dom.get(n, [])]
        kres["dominates"] = dom
        kres["front"] = front
        kres["front_dominating_M1"] = beats_m1
        res["k"][f"{k:g}"] = kres

    # MILP gap (k = 3)
    milp = {}
    for run in runs:
        p = CACHE / f"milp_{safe(run['name'])}_k3.npz"
        if not p.exists():
            continue
        z = np.load(p)
        # files from before the bound-rounding fix can have bound = incumbent + 1; a bound never exceeds it
        milp[run["name"]] = {"status": int(z["status"]), "incumbent": float(z["incumbent"]),
                             "bound": min(float(z["bound"]), float(z["incumbent"])), "secs": float(z["secs"])}
    per3 = res["k"]["3"]["per_run"]
    gap = {}
    for n in REMOVAL + [x + "+R" for x in REMOVAL]:
        exc, exc_rel, exc_b = {}, {}, {}
        for r, mi in milp.items():
            mm = per3[n].get(r)
            if not mm:
                continue
            explicit = mm["own"] + mm["finish"]
            nl = len(next(x for x in runs if x["name"] == r)["nu"])
            if mi["status"] == 0:
                exc[r] = (explicit - mi["incumbent"]) / nl
                exc_rel[r] = (explicit - mi["incumbent"]) / max(mi["incumbent"], 1)
            exc_b[r] = (explicit - mi["bound"]) / nl
        gap[n] = {"excess_share_optimal_runs": pooled(exc, grp), "excess_over_milp_optimal_runs":
                  pooled(exc_rel, grp), "excess_share_vs_bound_all_runs": pooled(exc_b, grp)}
    m1 = gap["M1"]
    res["milp"] = milp
    res["milp_gap"] = gap
    res["greedy_near_optimal"] = bool(
        m1["excess_share_optimal_runs"] and m1["excess_share_optimal_runs"]["mean"] <= 0.01
        and m1["excess_over_milp_optimal_runs"]["mean"] <= 0.25)

    # M5 scales on C2H4 (descriptive)
    c2 = next((x for x in runs if "C2H4" in x["name"]), None)
    if c2 is not None:
        try:
            sc = load_out(c2, "M5", 3.0)["extra"].get("scales", {})
            res["m5_c2h4_scales_vs_key"] = {s: [sc.get(s, 1.0), f] for s, f in C2H4_SCALE.items()}
            res["m5_c2h4_scales_top"] = dict(sorted(sc.items(), key=lambda kv: -kv[1])[:15])
        except FileNotFoundError:
            pass
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_text(res)


def fmt(p, d=3):
    return "—" if not p else f"{p['mean']:+.{d}f} [{p['ci95'][0]:+.{d}f}, {p['ci95'][1]:+.{d}f}]"


def write_text(res):
    L = []
    for k, kr in res["k"].items():
        L.append(f"=== k = {k}")
        L.append(f"{'method':10s} {'NLL':>8s} {'NLLcommon':>9s} {'kept':>7s} {'valid':>7s} {'infl':>7s} "
                 f"{'cov1':>5s} {'cov2':>5s} {'own':>6s} {'finish':>6s} {'cut':>6s} {'readm':>5s} {'swamp':>6s}")
        for n, v in kr["methods"].items():
            g = lambda mt, f="{:.3f}": f.format(v[mt]["mean"]) if v.get(mt) else "—"  # noqa: E731
            L.append(f"{n:10s} {g('nll'):>8s} {g('nll_common'):>9s} {g('kept', '{:.4f}'):>7s} "
                     f"{g('validated', '{:.4f}'):>7s} {g('inflated', '{:.0f}'):>7s} {g('cov1'):>5s} "
                     f"{g('cov2'):>5s} {g('own', '{:.0f}'):>6s} {g('finish', '{:.0f}'):>6s} "
                     f"{g('cut', '{:.0f}'):>6s} {g('readmitted', '{:.0f}'):>5s} {g('swamp', '{:.0f}'):>6s}")
        for base in ("M1", "M0"):
            L.append(f"-- vs {base}: dNLL / dNLLcommon / dkept / dvalidated")
            for n, v in kr[f"vs_{base}"].items():
                L.append(f"  {n:10s} {fmt(v['nll'])}  {fmt(v['nll_common'])}  {fmt(v['kept'], 4)}  "
                         f"{fmt(v['validated'], 4)}")
        L.append(f"front: {kr['front']}")
        L.append(f"front methods dominating M1: {kr['front_dominating_M1']}")
        L.append(f"dominance: {json.dumps(kr['dominates'])}")
    L.append("=== MILP (k = 3)")
    for r, mi in res["milp"].items():
        L.append(f"  {r[:34]:34s} status {mi['status']} incumbent {mi['incumbent']:.0f} bound "
                 f"{mi['bound']:.0f} ({mi['secs']:.0f} s)")
    for n, g in res["milp_gap"].items():
        L.append(f"  {n:10s} excess/lines (optimal runs) {fmt(g['excess_share_optimal_runs'], 4)}  "
                 f"excess/MILP {fmt(g['excess_over_milp_optimal_runs'])}  "
                 f"excess/lines vs bound (all) {fmt(g['excess_share_vs_bound_all_runs'], 4)}")
    L.append(f"greedy near-optimal (pre-registered): {res['greedy_near_optimal']}")
    if "m5_c2h4_scales_vs_key" in res:
        L.append(f"M5 C2H4 scales vs key: {res['m5_c2h4_scales_vs_key']}")
    text = "\n".join(L)
    (OUT / "results.txt").write_text(text + "\n")
    print(text)
