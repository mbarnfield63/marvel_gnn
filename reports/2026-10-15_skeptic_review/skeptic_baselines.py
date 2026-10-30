"""Skeptic experiments: deterministic baselines vs the shipped GNN checkpoint,
on exactly the held-out data train.py reports (same seeds, same splits)."""
import sys, json
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/mnt/c/Code/MARVEL_GNN/src")
from marvel_gnn.core.parse import parse_mrt_levels, parse_mrt_transitions
from marvel_gnn.core.solver import level_index, solve_energies
from marvel_gnn.core.uncertainty import marvel_solve, shortest_path_unc
from marvel_gnn.gnn import train as T
from marvel_gnn.gnn.data import ERROR_SCALE
from marvel_gnn.gnn.linkage import link_metrics
from marvel_gnn.gnn.labelfix import fix_metrics
from marvel_gnn.gnn.model import MarvelGNN

CO = Path("/mnt/c/Code/MARVEL_scraping/molecules/CO")
tr = parse_mrt_transitions(CO / "CO_isotopologues_all_input.txt")
pub = parse_mrt_levels(CO / "CO_isotopologues_all_output.txt")
out = {}


def reff_sigma(comp, assignments):
    """Exact WLS standard error of E_i - E_ground: sqrt(effective resistance)."""
    idx = level_index(comp)
    n = len(idx)
    lap = np.zeros((n, n))
    for t in comp:
        i, j, w = idx[t.upper], idx[t.lower], 1 / t.unc**2
        lap[i, i] += w; lap[j, j] += w; lap[i, j] -= w; lap[j, i] -= w
    g = np.zeros((n, n)); g[1:, 1:] = np.linalg.inv(lap[1:, 1:])
    e = solve_energies(comp)
    gr = idx[min(e, key=e.get)]
    r = np.array([g[idx[a], idx[a]] + g[gr, gr] - 2 * g[idx[a], gr] for a in assignments])
    chi2 = sum(((t.freq - (e[t.upper] - e[t.lower])) / t.unc) ** 2 for t in comp)
    birge = np.sqrt(chi2 / max(len(comp) - (n - 1), 1))
    return np.sqrt(np.clip(r, 0, None)) * ERROR_SCALE, birge


def scale_mle(shape, errors):
    """Closed-form MLE of a single global scale a for sigma = a * shape."""
    m = torch.isfinite(errors)
    s = torch.tensor(shape, dtype=torch.float32).unsqueeze(1).expand_as(errors)
    return float(torch.sqrt((errors[m] ** 2 / s[m] ** 2).mean()))


train = [T.prepare(i, tr, T.N_TRAIN_SAMPLES, seed=1) for i in T.TRAIN_ISOS]
val = [T.prepare(i, tr, T.N_VAL_SAMPLES, seed=2) for i in T.VAL_ISOS]

# fit one global scale per deterministic shape on TRAINING isotopologues only
fits = {}
for name in ["reff", "reff_birge", "dijkstra"]:
    errs, shapes = [], []
    for comp, g, e in train:
        keep = np.arange(len(g.assignments)) != g.ground
        r, b = reff_sigma(comp, g.assignments)
        sp = shortest_path_unc(comp, solve_energies(comp))
        shp = {"reff": r, "reff_birge": r * b,
               "dijkstra": np.array([sp[a] for a in g.assignments]) * ERROR_SCALE}[name]
        errs.append(e[keep]); shapes.append(shp[keep])
    fits[name] = scale_mle(np.concatenate(shapes), torch.cat(errs))
out["scale_fits"] = fits
print("global scales fitted on train isos:", fits)

model = MarvelGNN()
model.load_state_dict(torch.load("/mnt/c/Code/MARVEL_GNN/data/marvel_gnn.pt", map_location="cpu"))
model.eval()

for iso, (comp, g, errors) in zip(T.VAL_ISOS, val):
    keep = np.arange(len(g.assignments)) != g.ground
    err = errors[keep]
    with torch.no_grad():
        s_gnn = torch.exp(model.log_sigma(g)).numpy().astype(float)
        s_gnn2 = torch.exp(model.log_sigma(g)).numpy().astype(float)
    leg = marvel_solve(comp, bootstrap_iterations=100, rng=42)
    s_leg = np.array([leg[a][1] for a in g.assignments]) * ERROR_SCALE
    sp = shortest_path_unc(comp, solve_energies(comp))
    s_dij = np.array([sp[a] for a in g.assignments]) * ERROR_SCALE
    s_pub = np.array([pub[iso][a].unc for a in g.assignments]) * ERROR_SCALE
    r, b = reff_sigma(comp, g.assignments)
    rows = {"GNN (checkpoint)": s_gnn, "legacy Dijkstra+bootstrap": s_leg,
            "published e_E": s_pub, "Dijkstra only": s_dij,
            "WLS exact (R_eff), unscaled": r, "WLS exact x Birge": r * b,
            "Dijkstra x 1 fitted scalar": s_dij * fits["dijkstra"],
            "WLS exact x 1 fitted scalar": r * fits["reff"],
            "WLS x Birge x 1 fitted scalar": r * b * fits["reff_birge"]}
    res = {}
    for k, s in rows.items():
        nll, cov = T.fixed_sigma_metrics(np.maximum(s[keep], 1e-12), err)
        res[k] = (nll, cov)
    # Spearman between GNN and exact-WLS ranking
    from scipy.stats import spearmanr
    res["spearman_gnn_vs_reff"] = spearmanr(s_gnn[keep], r[keep]).statistic
    res["spearman_gnn_vs_dijkstra"] = spearmanr(s_gnn[keep], s_dij[keep]).statistic
    res["cpu_rerun_maxdiff"] = float(np.abs(s_gnn - s_gnn2).max())

    # leaf analysis: a degree-1 level's masked-refit error equals its parent's
    deg = {}
    for t in comp:
        deg[t.upper] = deg.get(t.upper, 0) + 1
        deg[t.lower] = deg.get(t.lower, 0) + 1
    idx = level_index(comp)
    E = errors.numpy()
    diffs, ratio_unc = [], []
    for t in comp:
        for leaf, par in ((t.upper, t.lower), (t.lower, t.upper)):
            if deg[leaf] == 1 and idx[par] != g.ground:
                a, b_ = E[idx[leaf]], E[idx[par]]
                m = np.isfinite(a) & np.isfinite(b_)
                if m.any():
                    diffs.append(np.abs(a[m] - b_[m]).max())
                    ratio_unc.append((s_gnn[idx[leaf]], s_gnn[idx[par]], t.unc * ERROR_SCALE))
    ru = np.array(ratio_unc)
    res["n_levels"] = int(len(g.assignments))
    res["n_leaves"] = len(diffs)
    res["leaf_target_maxdiff_vs_parent"] = float(max(diffs)) if diffs else None
    if len(ru):
        res["leaf_edge_unc_over_gnn_sigma_median"] = float(np.median(ru[:, 2] / ru[:, 0]))
        res["leaf_gnn_sigma_over_parent_median"] = float(np.median(ru[:, 0] / ru[:, 1]))
        res["leaf_frac_edge_unc_gt_gnn_sigma"] = float(np.mean(ru[:, 2] > ru[:, 0]))
    out[iso] = res
    print(iso, json.dumps(res, indent=1, default=float))

# outlier + label-correction baselines on the identical corrupted val copies
for i, (iso, (comp, g, e)) in enumerate(zip(T.VAL_ISOS, val)):
    graphs = T.prepare_corrupted(comp, T.N_VAL_CORRUPT, seed=200 + i, device="cpu")
    m_gnn = T.outlier_metrics(model, graphs)
    sc = {"raw": [], "stud": []}
    kinds = []
    for gg, labels, k, mags, fix in graphs:
        ea = gg.edge_attr[0::2].numpy()
        sc["raw"].append(ea[:, 2]); sc["stud"].append(ea[:, 4]); kinds.append(k)
    kinds = np.concatenate(kinds)
    res = {"gnn_ap": m_gnn["ap"], "gnn_ap_freq": m_gnn["ap_freq"], "gnn_ap_qn": m_gnn["ap_qn"],
           "base": m_gnn["base"]}
    for name, s in sc.items():
        s = np.concatenate(s)
        res[f"{name}_ap"] = T.average_precision(kinds > 0, s)
        for kind, kn in [(1, "freq"), (2, "qn")]:
            mk = (kinds == kind) | (kinds == 0)
            res[f"{name}_ap_{kn}"] = T.average_precision(kinds[mk] == kind, s[mk])
    # label correction: GNN vs argmin of the energy-frequency mismatch feature alone
    fm = fix_metrics(model, graphs)
    ranks = []
    for gg, *_, fix in graphs:
        if fix is None:
            continue
        mis = fix["extra"][:, :, 2].masked_fill(~fix["mask"], float("inf"))
        true = mis.gather(1, fix["target"].unsqueeze(1))
        ranks += (1 + (mis < true).sum(1)).tolist()
    ranks = np.array(ranks, float)
    res.update(fix_gnn_acc=fm["acc"], fix_gnn_mrr=fm["mrr"], fix_n=fm["n"],
               fix_rule_acc=float((ranks == 1).mean()), fix_rule_mrr=float((1 / ranks).mean()),
               fix_window=fm["n_cand"])
    # orphan linkage: GNN vs pure selection-rule ranking (|dJ|==1 first, ties random)
    links = T.prepare_linked(comp, T.N_VAL_LINK, seed=400 + i) if hasattr(T, "prepare_linked") else None
    lm = link_metrics(model, links)
    rk = []
    rng = np.random.default_rng(0)
    for mg, orphans in links:
        jm = mg.x[:, 1].numpy() * 50
        for og, oi, mi in orphans:
            jo = og.x[:, 1].numpy() * 50
            dj = np.abs(jo[:, None] - jm[None, :]).ravel()
            score = (np.abs(dj - 1) < 1e-6).astype(float) + 1e-3 * rng.random(dj.size)
            t = oi * len(jm) + mi
            rk.append(1 + (score > score[t]).sum())
    rk = np.array(rk, float)
    res.update(link_gnn_mrr=lm["mrr"], link_gnn_hits1=lm["hits1"], link_gnn_hits5=lm["hits5"],
               link_rule_mrr=float((1 / rk).mean()), link_rule_hits1=float((rk <= 1).mean()),
               link_rule_hits5=float((rk <= 5).mean()), link_pairs=lm["n_pairs"])
    out[iso + "_heads"] = res
    print(iso, json.dumps(res, indent=1, default=float))

json.dump(out, open(Path(__file__).with_name("skeptic_results.json"), "w"), indent=1, default=float)
