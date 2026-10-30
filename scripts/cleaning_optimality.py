"""Step 2: how optimal are the cleaning methods?

Pre-registered in reports/2026-10-30_cleaning_optimality/PREREGISTRATION.md.

Stages (each job is cached under data/dbmarvel/cleaning/, so reruns skip done work):
  uv run python scripts/cleaning_optimality.py prep            # per-run arrays + solver check
  uv run python scripts/cleaning_optimality.py jobs A|B|C|D    # print job lines for a phase
  uv run python scripts/cleaning_optimality.py run <job...>    # run one job
  uv run python scripts/cleaning_optimality.py report          # pool, write results
Phase X = every job as an independent chain (base method, its +R/M6
variants, their scoring) plus the MILP, for a Hypatia array. Phases: A base methods; B M6-1/M6-2 and +R variants (need A); C held-out scoring
(needs A+B); D the MILP. Run a phase with
  ... jobs A | xargs -P 6 -L 1 uv run python scripts/cleaning_optimality.py run
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # many small processes; no thread bloat

import json
import sys
import time
import zlib
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from marvel_gnn.core.parse import UNC_FLOOR  # noqa: E402
from marvel_gnn.core.selinv import selected_inverse  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))  # torch-free: dbmarvel_headroom only in prep/report
ROOT = Path(__file__).parents[1]

OUT = ROOT / "reports" / "2026-10-30_cleaning_optimality"
CACHE = ROOT / "data" / "dbmarvel" / "cleaning"
KS = (3.0, 2.0, 5.0)
BASE = ["M1", "M2", "M3a-H", "M3a-T", "M3b-H", "M3b-T", "M4", "M5"]
KFREE = ["M0", "M6-0"]                       # don't depend on k
PHASE_B = ["M6-1", "M6-2", "M1+R", "M2+R", "M3a-H+R", "M3a-T+R", "M4+R", "M5+R"]
REMOVAL = ["M1", "M2", "M3a-H", "M3a-T", "M4", "M5"]
N_FOLD = 10
MILP_LIMIT = 7200


# ---------------------------------------------------------------- data

def prep():
    """Write one npz per usable outlier run: line arrays over the unedited
    largest component, levels 0..n-1, plus M0's mask and used unc."""
    from dbmarvel_headroom import largest, records
    from marvel_gnn.core.solver import solve_energies
    CACHE.mkdir(parents=True, exist_ok=True)
    names = []
    for rec in records():
        if "outlier" not in rec["usable"]:
            continue
        tr = rec["trans"]
        comp = largest(tr, range(len(tr)))
        n_in = rec["neg"][comp] & ~rec["x"][comp]
        if n_in.sum() < 10:
            continue
        lines = [tr[i] for i in comp]
        lev = {}
        for t in lines:
            for a in (t.upper, t.lower):
                lev.setdefault(a, len(lev))
        run = {"ui": np.array([lev[t.upper] for t in lines]), "li": np.array([lev[t.lower] for t in lines]),
               "nu": np.array([t.freq for t in lines]),
               "u0": np.maximum([t.unc for t in lines], UNC_FLOOR),
               "used": np.maximum(rec["used"][comp], UNC_FLOOR),
               "neg": rec["neg"][comp], "x": rec["x"][comp], "n": len(lev),
               "src": np.array([t.ref.rsplit(".", 1)[0] for t in lines]),
               "fold": np.array([zlib.crc32(f"0:{t.ref}".encode()) % N_FOLD for t in lines])}
        # solver check against core.solver (same floored unc)
        ref = solve_energies(lines, list(run["u0"]))
        e_ref = np.array([ref[a] for a in lev])
        run["ground"] = int(np.argmin(e_ref))
        f = fit(run, np.ones(len(lines), bool), run["u0"], need_lu=True)
        e = f["E"] - np.nanmin(f["E"])
        # Deviation: the pre-registered 1e-8 cm-1 is below what two LU solvers
        # agree to at 1e4-1e5 cm-1 energies, and with weights spanning 1e14
        # (CaOH). Checked instead against each level's formal sigma.
        # Where they disagree (weights spanning 1e14, CaOH), the refined solve
        # must be the better LS minimum: chi2 no higher than core.solver's.
        sig = np.maximum(level_sigma(run, f), 1e-12)
        err = np.nanmax(np.abs(e - e_ref)[f["lev_in"]] / sig)
        w = 1 / run["u0"] ** 2
        chi2 = [float((w * (run["nu"] - (x[run["ui"]] - x[run["li"]])) ** 2).sum()) for x in (e, e_ref)]
        # Below 1e-7 cm-1 (a tenth of the unc floor) differences are float
        # noise: CO2 626's chi2 ~1e11 hits double precision at ~1e-8 relative.
        dabs = np.abs(e - e_ref)[f["lev_in"]]
        ok = np.all((dabs <= 1e-7) | (dabs / sig <= 0.01))
        assert ok or chi2[0] <= chi2[1] * (1 + 1e-7), (rec["name"], err, chi2)
        name = safe(rec["name"])
        np.savez(CACHE / f"run_{name}.npz", group=rec["group"], name=rec["name"], **run)
        names.append(name)
        print(f"{rec['name']}: {len(lines)} lines, {len(lev)} levels, max |dE|/sigma {err:.1e}, "
              f"chi2 ours/core {chi2[0]:.6g}/{chi2[1]:.6g}", flush=True)
    (CACHE / "runs.json").write_text(json.dumps(names))


def safe(name):
    return name.replace(":", "_").replace("@", "_")


def load_run(name):
    z = np.load(CACHE / f"run_{name}.npz")
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def run_names():
    return json.loads((CACHE / "runs.json").read_text())


# ---------------------------------------------------------------- solver

def fit(run, keep, u, need_lu=False):
    """LS solve on the largest component of the kept lines (levels counted).
    Returns E (nan off-component), lines_in (kept lines in it), lev_in, and
    optionally the symmetric-mode LU of the reduced system (pinned level)."""
    ui, li, n = run["ui"], run["li"], run["n"]
    k = np.flatnonzero(keep)
    adj = coo_matrix((np.ones(len(k)), (ui[k], li[k])), shape=(n, n))
    ncomp, lab = connected_components(adj, directed=False)
    touched = np.zeros(n, bool)
    touched[ui[k]] = touched[li[k]] = True
    sizes = np.bincount(lab[touched], minlength=ncomp)
    lev_in = touched & (lab == np.argmax(sizes))
    lines_in = keep & lev_in[ui]
    k = np.flatnonzero(lines_in)
    levels = np.flatnonzero(lev_in)
    pin = run.get("ground", levels[0])
    if not lev_in[pin]:
        pin = levels[0]
    pos = np.full(n, -1)
    rest = levels[levels != pin]
    pos[rest] = np.arange(len(rest))
    w = 1.0 / u[k] ** 2
    a, b = pos[ui[k]], pos[li[k]]
    rows, cols, vals = [], [], []
    for p, q, s in ((a, a, w), (b, b, w), (a, b, -w), (b, a, -w)):
        ok = (p >= 0) & (q >= 0)
        rows.append(p[ok]), cols.append(q[ok]), vals.append(s[ok])
    m = len(rest)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(m, m)).tocsc()
    y = np.zeros(m)
    np.add.at(y, a[a >= 0], (run["nu"][k] * w)[a >= 0])
    np.add.at(y, b[b >= 0], -(run["nu"][k] * w)[b >= 0])
    lu = splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options={"SymmetricMode": True}) if m else None
    E = np.full(n, np.nan)
    E[pin] = 0.0
    if m:
        x = lu.solve(y)
        x += lu.solve(y - A @ x)  # one refinement step: weights span up to 1e14
        E[rest] = x
    out = {"E": E, "lines_in": lines_in, "lev_in": lev_in, "pos": pos}
    if need_lu:
        out["lu"] = lu
    return out


def zscore(run, f, u):
    """|r|/u for every line whose two levels are on the component (nan otherwise)."""
    r = run["nu"] - (f["E"][run["ui"]] - f["E"][run["li"]])
    return np.abs(r) / u


def selinv(f):
    """Selected inverse of the reduced system: (diag per level, Z, perm)."""
    lu = f["lu"]
    assert (lu.perm_r == lu.perm_c).all()
    lower = lu.L.tocsc()
    lower.sort_indices()
    return selected_inverse(lower, lu.U.diagonal()), lu.perm_c


def leverage(run, f, u):
    """h for the component's lines (0 elsewhere), as gnn.data.edge_leverages."""
    Z, perm = selinv(f)
    k = np.flatnonzero(f["lines_in"])
    p = np.where(f["pos"] >= 0, perm[np.maximum(f["pos"], 0)], -1)
    pu, pl = p[run["ui"][k]], p[run["li"][k]]
    lo, hi = np.minimum(pu, pl), np.maximum(pu, pl)
    zd = np.r_[Z.diagonal(), 0.0]
    g = np.zeros(len(k))
    ok = lo >= 0
    g[ok] = np.asarray(Z[hi[ok], lo[ok]]).ravel()
    h = np.zeros(len(run["nu"]))
    h[k] = np.clip((zd[pu] + zd[pl] - 2 * g) / u[k] ** 2, 0, 1)
    return h


def level_sigma(run, f):
    """Formal level sigma sqrt(G_jj) relative to the pinned level, per component level."""
    Z, perm = selinv(f)
    d = Z.diagonal()
    pos = f["pos"]
    s = np.where(pos >= 0, np.sqrt(np.abs(d[perm[np.maximum(pos, 0)]])), 0.0)
    return s[f["lev_in"]]


def pred_var(run, f, idx):
    """sigma^2 of E_u - E_l for lines idx (both levels on the component), by
    batched solves with the component's LU."""
    pos = f["pos"]
    out = np.empty(len(idx))
    for s in range(0, len(idx), 400):
        part = idx[s:s + 400]
        R = np.zeros((f["lu"].shape[0], len(part)))
        cu, cl = pos[run["ui"][part]], pos[run["li"][part]]
        j = np.arange(len(part))
        R[cu[cu >= 0], j[cu >= 0]] += 1
        R[cl[cl >= 0], j[cl >= 0]] -= 1
        X = f["lu"].solve(R)
        out[s:s + 400] = (R * X).sum(0)
    return out


# ---------------------------------------------------------------- methods

class Track:
    """Removal attribution: own rule, finish step, cut off; plus re-admitted."""

    def __init__(self, m):
        self.why = np.zeros(m, int)  # 0 kept, 1 own, 2 finish, 3 cut off

    def cut(self, before, after, tag=3):
        self.why[before & ~after & (self.why == 0)] = tag


def greedy(run, keep, u, k, tr, tag, stud=False):
    """Remove the max-|r|/u (or max studentized) line and re-solve until max
    z <= k. Studentized: lines with h >= 0.999 are never picked; the loop
    stops if only such lines violate (a finish M1 then runs)."""
    keep = keep.copy()
    while True:
        f = fit(run, keep, u, need_lu=stud)
        tr.cut(keep, f["lines_in"])
        keep = f["lines_in"]
        z = np.where(keep, zscore(run, f, u), 0.0)
        if z.max() <= k:
            return keep
        if stud:
            h = leverage(run, f, u)
            s = np.where(keep & (h < 0.999), z / np.sqrt(np.maximum(1 - h, 1e-12)), -1.0)
            if s.max() <= k:
                return keep
            worst = int(np.argmax(s))
        else:
            worst = int(np.argmax(z))
        keep[worst] = False
        tr.why[worst] = tag


def irls(run, keep, u, c, kind, w0=None):
    """Huber/Tukey IRLS on |r|/u. Returns robust weights (0 = out) and z."""
    w = np.where(keep, 1.0, 0.0) if w0 is None else w0.copy()
    e_old = None
    tol = 1e-3 * u[keep].min()
    for _ in range(100):
        act = keep & (w > 0)
        f = fit(run, act, u / np.sqrt(np.where(w > 0, w, 1.0)))
        z = zscore(run, f, u)
        on = keep & f["lev_in"][run["ui"]] & f["lev_in"][run["li"]]
        zz = np.where(on, z, np.inf)
        w = np.where(on, np.minimum(1.0, c / np.maximum(zz, 1e-300)) if kind == "H" else
                     np.where(zz < c, (1 - (zz / c) ** 2) ** 2, 0.0), 0.0)
        e = f["E"]
        if e_old is not None and np.nanmax(np.abs(e - e_old)) < tol:
            break
        e_old = e
    return w, np.where(on, z, np.inf)


def l1(run, keep, u):
    """Weighted L1: min sum |r|/u over energies (HiGHS). Returns z_L1 (inf off-component)."""
    f = fit(run, keep, u)
    keep = f["lines_in"]
    k = np.flatnonzero(keep)
    pos = f["pos"]
    m, nl = len(k), int((pos >= 0).sum())
    a, b = pos[run["ui"][k]], pos[run["li"][k]]
    j = np.arange(m)
    # variables: x (nl reduced energies), t (m); rows: +-(nu - (x_a - x_b))/u <= t
    def block(sign):
        rows = np.r_[j[a >= 0], j[b >= 0], j]
        cols = np.r_[a[a >= 0], b[b >= 0], nl + j]
        vals = np.r_[-sign / u[k][a >= 0], sign / u[k][b >= 0], -np.ones(m)]
        return coo_matrix((vals, (rows, cols)), shape=(m, nl + m))
    A = vstack([block(1.0), block(-1.0)]).tocsr()
    rhs = np.r_[-run["nu"][k] / u[k], run["nu"][k] / u[k]]
    res = linprog(np.r_[np.zeros(nl), np.ones(m)], A_ub=A, b_ub=rhs,
                  bounds=[(None, None)] * nl + [(0, None)] * m, method="highs")
    assert res.status == 0, res.message
    E = np.full(run["n"], np.nan)
    E[f["lev_in"] & (pos < 0)] = 0.0
    E[pos >= 0] = res.x[pos[pos >= 0]]
    r = np.abs(run["nu"] - (E[run["ui"]] - E[run["li"]])) / u
    return np.where(keep, r, np.inf)


def helmert(run, keep, u0, k):
    """Huber IRLS (c = k) alternated with per-source scale updates."""
    src = run["src"]
    names = np.unique(src)
    s = {n: 1.0 for n in names}
    for _ in range(20):
        u = u0 * np.array([s[x] for x in src])
        w, z = irls(run, keep, u, k, "H")
        act = keep & (w > 0)
        f = fit(run, act, u / np.sqrt(np.where(w > 0, w, 1.0)), need_lu=True)
        h = leverage(run, f, u / np.sqrt(np.where(w > 0, w, 1.0)))
        change = 0.0
        for n in names:
            m = act & (src == n)
            red = (w[m] * (1 - h[m])).sum()
            if red >= 5:
                new = float(np.clip(s[n] * np.sqrt((w[m] * z[m] ** 2).sum() / red), 0.5, 100))
                change = max(change, abs(new - s[n]) / s[n])
                s[n] = new
        if change < 0.01:
            break
    u = u0 * np.array([s[x] for x in src])
    w, z = irls(run, keep, u, k, "H")
    return u, z, s


def replay(run, keep, u):
    u = u.copy()
    for _ in range(100):
        f = fit(run, keep, u)
        keep = f["lines_in"]
        r = np.where(keep, np.abs(run["nu"] - (f["E"][run["ui"]] - f["E"][run["li"]])), 0.0)
        new = np.where(keep, np.maximum(u, r), u)
        done = (new / u - 1).max() <= 1e-3
        u = new
        if done:
            break
    return keep, u


def readmit(run, keep, u, k, tr):
    """+R: re-add removed lines (both levels on the network), smallest
    predicted z first, keeping each only if max z <= k still holds."""
    keep = keep.copy()
    tried = np.zeros(len(keep), bool)
    added = 0
    while True:
        f = fit(run, keep, u)
        z = zscore(run, f, u)
        cand = ~keep & ~tried & f["lev_in"][run["ui"]] & f["lev_in"][run["li"]] & (z <= k)
        if not cand.any():
            break
        i = int(np.flatnonzero(cand)[np.argmin(z[cand])])
        tried[i] = True
        trial = keep.copy()
        trial[i] = True
        g = fit(run, trial, u)
        if g["lines_in"][i] and np.nanmax(np.where(g["lines_in"], zscore(run, g, u), 0)) <= k:
            keep = g["lines_in"]
            tr.why[i] = 0
            added += 1
    return keep, added


def method(run, name, k):
    """-> dict(keep, u, why, extra) for one method at threshold k."""
    m = len(run["nu"])
    u0 = run["u0"]
    tr = Track(m)
    all_on = np.ones(m, bool)
    extra = {}
    if name == "M0":
        keep = ~run["neg"]
        f = fit(run, keep, run["used"])
        tr.why[~keep] = 1
        tr.cut(keep, f["lines_in"])
        return {"keep": f["lines_in"], "u": run["used"], "why": tr.why, "extra": extra}
    if name == "M1":
        keep = greedy(run, all_on, u0, k, tr, 1)
        return {"keep": keep, "u": u0, "why": tr.why, "extra": extra}
    if name == "M2":
        keep = greedy(run, all_on, u0, k, tr, 1, stud=True)
        keep = greedy(run, keep, u0, k, tr, 2)
        return {"keep": keep, "u": u0, "why": tr.why, "extra": extra}
    if name in ("M3a-H", "M3a-T", "M3b-H", "M3b-T"):
        kind = name[-1]
        w, z = irls(run, all_on, u0, k, "H")
        if kind == "T":
            w, z = irls(run, all_on, u0, 2 * k, "T", w0=w)
        if name.startswith("M3a"):
            keep = z <= k
            tr.why[~keep] = 1
            keep = greedy(run, keep, u0, k, tr, 2)
            return {"keep": keep, "u": u0, "why": tr.why, "extra": extra}
        keep = w > 0
        tr.why[~keep] = 1
        u = u0 / np.sqrt(np.where(keep, w, 1.0))
        f = fit(run, keep, u)
        tr.cut(keep, f["lines_in"])
        return {"keep": f["lines_in"], "u": u, "why": tr.why, "extra": extra}
    if name == "M4":
        z = l1(run, all_on, u0)
        keep = z <= k
        tr.why[~keep] = 1
        keep = greedy(run, keep, u0, k, tr, 2)
        return {"keep": keep, "u": u0, "why": tr.why, "extra": extra}
    if name == "M5":
        u, z, s = helmert(run, all_on, u0, k)
        keep = z <= k
        tr.why[~keep] = 1
        keep = greedy(run, keep, u, k, tr, 2)
        extra["scales"] = {n: v for n, v in s.items() if v != 1.0}
        return {"keep": keep, "u": u, "why": tr.why, "extra": extra}
    if name == "M6-0":
        keep, u = replay(run, all_on, u0)
        tr.cut(all_on, keep)
        return {"keep": keep, "u": u, "why": tr.why, "extra": extra}
    if name in ("M6-1", "M6-2"):
        base = load_out(run, "M" + name[-1], k)
        keep, u = replay(run, base["keep"], base["u"])
        return {"keep": keep, "u": u, "why": base["why"], "extra": extra}
    if name.endswith("+R"):
        base = load_out(run, name[:-2], k)
        tr.why = base["why"].copy()
        keep, added = readmit(run, base["keep"], base["u"], k, tr)
        extra["readmitted"] = added
        return {"keep": keep, "u": base["u"], "why": tr.why, "extra": extra}
    raise ValueError(name)


def out_path(rname, name, k):
    kk = "x" if name in KFREE else f"{k:g}"
    return CACHE / f"out_{rname}_{name}_k{kk}.npz"


def load_out(run, name, k):
    z = np.load(out_path(safe(run["name"]), name, k), allow_pickle=True)
    return {"keep": z["keep"], "u": z["u"], "why": z["why"], "extra": z["extra"].item(),
            "secs": float(z["secs"])}


def run_method(rname, name, k):
    p = out_path(rname, name, k)
    if p.exists():
        return
    run = load_run(rname)
    t0 = time.time()
    o = method(run, name, k)
    # swamping: removed lines that fit the final network within k
    f = fit(run, o["keep"], o["u"])
    z = zscore(run, f, o["u"])
    on = f["lev_in"][run["ui"]] & f["lev_in"][run["li"]]
    o["extra"]["swamp"] = int((~o["keep"] & on & (z <= k)).sum())
    np.savez(p, keep=o["keep"], u=o["u"], why=o["why"], extra=np.array(o["extra"], dtype=object),
             secs=time.time() - t0)
    print(f"{rname} {name} k={k:g}: kept {o['keep'].mean():.4f} in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- scoring

def score(rname, name, k):
    """Held-out CV + network metrics for one method output."""
    p = CACHE / f"score_{rname}_{name}_k{'x' if name in KFREE else f'{k:g}'}.npz"
    if p.exists():
        return
    run = load_run(rname)
    o = load_out(run, name, k)
    keep, u = o["keep"], o["u"]
    t0 = time.time()
    f = fit(run, keep, u, need_lu=True)
    keep = f["lines_in"]
    sig = level_sigma(run, f)
    g = nx.Graph()
    g.add_edges_from(zip(run["ui"][keep], run["li"][keep]))
    br = nx.bridges(g)
    nb = {}
    for a, b in br:
        nb[a] = nb.get(a, 0) + 1
        nb[b] = nb.get(b, 0) + 1
    validated = sum(g.degree(v) > nb.get(v, 0) for v in g.nodes)
    zfin = np.where(keep, zscore(run, f, u), 0.0)
    nll = np.full(len(keep), np.nan)
    zz = np.full(len(keep), np.nan)
    disc = 0
    for fold in range(N_FOLD):
        held = keep & (run["fold"] == fold)
        tf = fit(run, keep & (run["fold"] != fold), u, need_lu=True)
        on = held & tf["lev_in"][run["ui"]] & tf["lev_in"][run["li"]]
        disc += int((held & ~on).sum())
        idx = np.flatnonzero(on)
        if not len(idx):
            continue
        pred = tf["E"][run["ui"][idx]] - tf["E"][run["li"][idx]]
        s2 = u[idx] ** 2 + pred_var(run, tf, idx)
        zz[idx] = (run["nu"][idx] - pred) / np.sqrt(s2)
        nll[idx] = 0.5 * np.log(2 * np.pi * s2) + 0.5 * zz[idx] ** 2
    np.savez(p, keep=keep, nll=nll, z=zz, disc=disc, validated=validated, levels=int(f["lev_in"].sum()),
             sigma_med=float(np.median(sig)), maxz=float(zfin.max()), secs=time.time() - t0)
    print(f"{rname} {name} k={k:g}: NLL {np.nanmean(nll):.3f} ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------- MILP

def run_milp(rname, k=3.0, tag="milp"):
    p = CACHE / f"{tag}_{rname}_k{k:g}.npz"
    if p.exists():
        return
    run = load_run(rname)
    u0, nu, ui, li, n = run["u0"], run["nu"], run["ui"], run["li"], run["n"]
    m = len(nu)
    f = fit(run, np.ones(m, bool), u0)
    S = np.nanmax(f["E"]) - np.nanmin(f["E"]) + 1.0
    M = nu + 2 * S
    r = np.arange(m)
    blocks = [coo_matrix((np.r_[s * np.ones(m), -s * np.ones(m), -M], (np.r_[r, r, r], np.r_[ui, li, n + r])),
                         shape=(m, n + m)) for s in (1.0, -1.0)]
    A = vstack(blocks).tocsr()
    ub = np.r_[nu + k * u0, -nu + k * u0]
    lb_, ub_ = np.r_[np.full(n, -S), np.zeros(m)], np.r_[np.full(n, 2 * S), np.ones(m)]
    lb_[run["ground"]] = ub_[run["ground"]] = 0.0
    t0 = time.time()
    res = milp(np.r_[np.zeros(n), np.ones(m)], constraints=LinearConstraint(A, -np.inf, ub),
               integrality=np.r_[np.zeros(n), np.ones(m)], bounds=Bounds(lb_, ub_),
               options={"time_limit": MILP_LIMIT, "mip_rel_gap": 0})
    inc = float(np.round(res.fun)) if res.x is not None else np.nan
    raw = getattr(res, "mip_dual_bound", None)
    raw = np.nan if raw is None else float(raw)  # None when HiGHS stops without a bound
    bound = float(np.ceil(raw - 1e-3))  # HiGHS reports e.g. 9.0000035 for an optimum of 9
    removed = res.x[n:] > 0.5 if res.x is not None else np.zeros(m, bool)
    np.savez(p, status=res.status, message=res.message, incumbent=inc, bound=bound, bound_raw=raw,
             removed=removed, E=res.x[:n] if res.x is not None else np.full(n, np.nan), secs=time.time() - t0)
    print(f"{rname} MILP: {res.message[:40]} incumbent {inc} bound {bound} ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------- driver

def jobs(phase):
    """Print the not-yet-cached job lines of a phase (one per line)."""
    for line in _jobs(phase):
        kind, r, *rest = line.split()
        k = float(rest[-1])
        done = (out_path(r, rest[0], k) if kind in ("method", "chain") else
                CACHE / f"score_{r}_{rest[0]}_k{'x' if rest[0] in KFREE else f'{k:g}'}.npz"
                if kind == "score" else CACHE / f"milp_{r}_k{k:g}.npz")
        if kind == "chain":
            names = [rest[0]] + CHAIN.get(rest[0], [])
            kk = lambda nm: "x" if nm in KFREE else f"{k:g}"  # noqa: E731
            done = all((CACHE / f"score_{r}_{nm}_k{kk(nm)}.npz").exists() for nm in names)
            if not done:
                print(line)
        elif not done.exists():
            print(line)


def _jobs(phase):
    names = run_names()
    for r in names:
        if phase == "A":
            for nm in KFREE:
                yield (f"method {r} {nm} 3")
            for k in KS:
                for nm in BASE:
                    yield (f"method {r} {nm} {k:g}")
        elif phase == "B":
            for k in KS:
                for nm in PHASE_B:
                    yield (f"method {r} {nm} {k:g}")
        elif phase == "C":
            for nm in KFREE:
                yield (f"score {r} {nm} 3")
            for k in KS:
                for nm in BASE + PHASE_B:
                    yield (f"score {r} {nm} {k:g}")
        elif phase == "D":
            yield (f"milp {r} 3")
        elif phase == "X":  # everything, as independent chains (Hypatia array)
            for nm in KFREE:
                yield f"chain {r} {nm} 3"
            for k in KS:
                for nm in BASE:
                    yield f"chain {r} {nm} {k:g}"
            yield f"milp {r} 3"


CHAIN = {"M1": ["M1+R", "M6-1"], "M2": ["M2+R", "M6-2"], "M3a-H": ["M3a-H+R"], "M3a-T": ["M3a-T+R"],
         "M4": ["M4+R"], "M5": ["M5+R"]}


def run_chain(r, name, k):
    """Base method, its dependent variants, then the held-out scoring of each."""
    for nm in [name] + CHAIN.get(name, []):
        run_method(r, nm, k)
    for nm in [name] + CHAIN.get(name, []):
        score(r, nm, k)


def main():
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "jobs":
        jobs(sys.argv[2])
    elif cmd == "run":
        kind, rname, k = sys.argv[2], sys.argv[3], float(sys.argv[-1])
        if kind == "method":
            run_method(rname, sys.argv[4], k)
        elif kind == "score":
            score(rname, sys.argv[4], k)
        elif kind == "milp":
            run_milp(rname, k)
        elif kind == "chain":
            run_chain(rname, sys.argv[4], k)
    elif cmd == "report":
        from cleaning_report import report
        report()


if __name__ == "__main__":
    main()
