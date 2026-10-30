"""Step 2 post hoc (not pre-registered): is greedy's gap to the MILP real under
greedy's own criterion?

The MILP asks only that *some* energies put every kept line within k*u0; M1
asks that the LS energies do. So for each MILP-optimal run: take the MILP's
kept set, LS-solve it, then finish with M1 until LS max z <= k ("MILP+F").
MILP+F is a feasible M1-style answer; its explicit removals vs M1's are the
gap greedy actually leaves. MILP+F is then scored like any method.

Also: the MILP's own energies are checked against its constraints (big-M with
HiGHS's integrality tolerance can leave M*1e-6 of slack on a "kept" line).

Run: uv run python scripts/cleaning_posthoc.py run <run>    (one run)
     uv run python scripts/cleaning_posthoc.py verify <run> (MILP rerun, saves E)
     uv run python scripts/cleaning_posthoc.py summary
     uv run python scripts/cleaning_posthoc.py nested <run> | nested-summary

Nested check (VaultAgent review): the pre-registered score gives a held-out
line the method's final u, which for inflating methods was set from that
line's own full-network residual (replay: u_i >= |r_i|). Here the inflation is
redone per fold on the training lines only, and held-out lines get (i) u0 or
(ii) u0 x the median training inflation factor of their source.
"""

import json
import sys
import time

import numpy as np

from cleaning_optimality import (CACHE, OUT, Track, fit, greedy, irls, load_out, load_run, out_path, pred_var,
                                 replay, run_milp, run_names, safe, score, zscore)

K = 3.0
NAME = "MILP+F"


def optimal_runs():
    rs = []
    for r in run_names():
        p = CACHE / f"milp_{r}_k3.npz"
        if p.exists() and int(np.load(p)["status"]) == 0:
            rs.append(r)
    return rs


def run(rname):
    p = out_path(rname, NAME, K)
    if not p.exists():
        run_milp(rname, K, tag="milpx")
        z = np.load(CACHE / f"milpx_{rname}_k3.npz")
        if int(z["status"]) != 0:  # HiGHS is not reproducible on every run (16O-13C-18O: solve error)
            print(f"{rname}: MILP rerun status {int(z['status'])}, skipped")
            return
        r = load_run(rname)
        u0, removed = r["u0"], z["removed"]
        t0 = time.time()
        f = fit(r, ~removed, u0)
        zz = np.where(f["lines_in"], zscore(r, f, u0), 0.0)
        tr = Track(len(removed))
        tr.why[removed] = 1
        tr.cut(~removed, f["lines_in"])
        keep = greedy(r, f["lines_in"], u0, K, tr, 2)
        extra = {"milp_removed": int(removed.sum()), "ls_maxz_on_milp_set": float(zz.max()),
                 "ls_viol_on_milp_set": int((zz > K).sum())}
        np.savez(p, keep=keep, u=u0, why=tr.why, extra=np.array(extra, dtype=object), secs=time.time() - t0)
    score(rname, NAME, K)


def verify(rname):
    """Big-M check: lines the MILP keeps but its own energies put above k*u0.
    removed + violators is a truly feasible set (an upper bound on the optimum)."""
    run_milp(rname, K, tag="milpe")
    z = np.load(CACHE / f"milpe_{rname}_k3.npz")
    r = load_run(rname)
    zz = np.abs(r["nu"] - (z["E"][r["ui"]] - z["E"][r["li"]])) / r["u0"]
    viol = ~z["removed"] & (zz > K * (1 + 1e-6))
    return {"status": int(z["status"]), "removed": int(z["removed"].sum()), "violators": int(viol.sum()),
            "max_viol_cm": float(np.max(np.abs(r["nu"] - (z["E"][r["ui"]] - z["E"][r["li"]]))[viol]
                                        - K * r["u0"][viol])) if viol.any() else 0.0}


NESTED = ["M1", "M6-1", "M6-2", "M3b-T", "M0"]


def train_u(r, name, train):
    """Inflation redone on the training lines only, from u0 (removal set fixed)."""
    u0 = r["u0"]
    if name == "M1":
        return train, u0
    if name in ("M6-1", "M6-2", "M0"):  # M0: the curated set, replayed from u0 (MARVEL 4 stand-in)
        return replay(r, train, u0)
    w, _ = irls(r, train, u0, K, "H")  # M3b-T
    w, _ = irls(r, train, u0, 2 * K, "T", w0=w)
    keep = w > 0
    return keep, u0 / np.sqrt(np.where(keep, w, 1.0))


def nested(rname):
    p = CACHE / f"nested_{rname}_k3.npz"
    if p.exists():
        return
    r = load_run(rname)
    u0, src = r["u0"], r["src"]
    t0 = time.time()
    res = {}
    for name in NESTED:
        keep = load_out(r, name, K)["keep"]
        nll = np.full((2, len(keep)), np.nan)
        zz = np.full((2, len(keep)), np.nan)
        for fold in range(10):
            held = keep & (r["fold"] == fold)
            tk, tu = train_u(r, name, keep & (r["fold"] != fold))
            tf = fit(r, tk, tu, need_lu=True)
            tk = tf["lines_in"]
            idx = np.flatnonzero(held & tf["lev_in"][r["ui"]] & tf["lev_in"][r["li"]])
            if not len(idx):
                continue
            pred = r["nu"][idx] - (tf["E"][r["ui"][idx]] - tf["E"][r["li"][idx]])
            pv = pred_var(r, tf, idx)
            fac = np.ones(len(idx))
            ratio = tu / u0
            for x in np.unique(src[idx]):
                tr = tk & (src == x)
                if tr.any():
                    fac[src[idx] == x] = np.median(ratio[tr])
            for v, uh in enumerate((u0[idx], u0[idx] * fac)):
                s2 = uh ** 2 + pv
                zz[v, idx] = pred / np.sqrt(s2)
                nll[v, idx] = 0.5 * np.log(2 * np.pi * s2) + 0.5 * zz[v, idx] ** 2
        res[f"{name}_nll"], res[f"{name}_z"] = nll, zz
        print(f"{rname} {name}: NLL u0 {np.nanmean(nll[0]):.3f} src-median {np.nanmean(nll[1]):.3f}", flush=True)
    np.savez(p, secs=time.time() - t0, **res)


def nested_summary():
    from cleaning_report import dominates, groups_of, pooled
    runs = [load_run(x) for x in run_names()]
    grp = groups_of(runs)
    per3 = json.loads((OUT / "results.json").read_text())["k"]["3"]["per_run"]
    rows = {v: {n: {} for n in NESTED} for v in (0, 1)}
    for r in runs:
        p = CACHE / f"nested_{safe(r['name'])}_k3.npz"
        if not p.exists():
            continue
        z = np.load(p)
        common = np.ones(len(r["nu"]), bool)
        for n in NESTED:
            common &= ~np.isnan(z[f"{n}_nll"][0])
        for v in (0, 1):
            for n in NESTED:
                nl, zv = z[f"{n}_nll"][v], z[f"{n}_z"][v]
                ok = ~np.isnan(nl)
                rows[v][n][r["name"]] = {"nll": float(nl[ok].mean()), "nll_common": float(nl[common].mean()),
                                         "cov1": float((np.abs(zv[ok]) <= 1).mean()),
                                         "cov2": float((np.abs(zv[ok]) <= 2).mean()),
                                         "kept": per3[n][r["name"]]["kept"],
                                         "validated": per3[n][r["name"]]["validated"]}
    out = {}
    L = []
    c2 = next(x["name"] for x in runs if "C2H4" in x["name"])
    for v, lab in ((0, "held-out u = u0"), (1, "held-out u = u0 x source median training factor")):
        R = rows[v]
        o = {"methods": {}, "vs_M1": {}, "c2h4": {}, "dominates": {}}
        L.append(f"=== {lab}")
        L.append(f"{'method':7s} {'NLL':>7s} {'common':>7s} {'cov1':>6s} {'cov2':>6s}  dNLL vs M1 [CI]   dcommon vs M1 [CI]"
                 f"   C2H4 NLL/common")
        for n in NESTED:
            pm = {mt: pooled({k_: x[mt] for k_, x in R[n].items()}, grp) for mt in ("nll", "nll_common", "cov1", "cov2")}
            d = {mt: pooled({k_: R[n][k_][mt] - R["M1"][k_][mt] for k_ in R[n]}, grp)
                 for mt in ("nll", "nll_common", "kept", "validated")}
            o["methods"][n], o["vs_M1"][n] = pm, d
            o["c2h4"][n] = {mt: R[n][c2][mt] for mt in ("nll", "nll_common", "cov1", "cov2")}
            L.append(f"{n:7s} {pm['nll']['mean']:7.3f} {pm['nll_common']['mean']:7.3f} {pm['cov1']['mean']:6.3f} "
                     f"{pm['cov2']['mean']:6.3f}  {d['nll']['mean']:+.3f} [{d['nll']['ci95'][0]:+.3f}, "
                     f"{d['nll']['ci95'][1]:+.3f}]  {d['nll_common']['mean']:+.3f} [{d['nll_common']['ci95'][0]:+.3f}, "
                     f"{d['nll_common']['ci95'][1]:+.3f}]  {o['c2h4'][n]['nll']:.3f}/{o['c2h4'][n]['nll_common']:.3f}")
        for a_ in NESTED:
            for b_ in NESTED:
                if a_ != b_:
                    dd = {mt: pooled({k_: R[a_][k_][mt] - R[b_][k_][mt] for k_ in R[a_]}, grp)
                          for mt in ("nll", "kept", "validated")}
                    if dominates(dd):
                        o["dominates"].setdefault(a_, []).append(b_)
        L.append(f"dominance: {json.dumps(o['dominates'])}")
        out[lab] = o
    (OUT / "posthoc_nested.json").write_text(json.dumps(out, indent=1))
    text = "\n".join(L)
    (OUT / "posthoc_nested.txt").write_text(text + "\n")
    print(text)


def summary():
    res = json.loads((OUT / "results.json").read_text())["k"]["3"]["per_run"]
    rows = {}
    for rname in optimal_runs():
        p = out_path(rname, NAME, K)
        if not p.exists():
            continue
        r = load_run(rname)
        o = np.load(p, allow_pickle=True)
        s = np.load(CACHE / f"score_{rname}_{NAME}_k3.npz")
        ex, why = o["extra"].item(), o["why"]
        m1 = res["M1"][r["name"]]
        ver = verify(rname) if (CACHE / f"milpe_{rname}_k3.npz").exists() else {}
        rows[r["name"]] = {"verify": ver,
            **ex, "milpf_explicit": int(((why == 1) | (why == 2)).sum()), "milpf_finish": int((why == 2).sum()),
            "m1_explicit": m1["own"] + m1["finish"], "n_lines": len(r["nu"]),
            "milpf_kept": float(o["keep"].mean()), "m1_kept": m1["kept"],
            "milpf_nll": float(np.nanmean(s["nll"])), "m1_nll": m1["nll"],
            "milpf_validated": float(s["validated"] / r["n"]), "m1_validated": m1["validated"]}
    (OUT / "posthoc_milp_finish.json").write_text(json.dumps(rows, indent=1))
    for n, v in rows.items():
        print(f"{n[:30]:30s} lines {v['n_lines']:6d} MILP {v['milp_removed']:4d} LS-viol {v['ls_viol_on_milp_set']:4d} "
              f"(maxz {v['ls_maxz_on_milp_set']:.1f})  MILP+F {v['milpf_explicit']:5d}  M1 {v['m1_explicit']:5d}  "
              f"NLL {v['milpf_nll']:.3f} vs {v['m1_nll']:.3f}  valid {v['milpf_validated']:.4f} vs "
              f"{v['m1_validated']:.4f}  verify {v['verify']}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(safe(sys.argv[2]) if ":" in sys.argv[2] else sys.argv[2])
    elif sys.argv[1] == "verify":
        print(verify(sys.argv[2]))
    elif sys.argv[1] == "nested":
        nested(sys.argv[2])
    elif sys.argv[1] == "nested-summary":
        nested_summary()
    elif sys.argv[1] == "list":
        print("\n".join(optimal_runs()))
    else:
        summary()
