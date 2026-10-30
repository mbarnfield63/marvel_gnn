"""Blind C2H4 test: POST HOC controls (not pre-registered). Run after c2h4_blind_score.py."""
import numpy as np
from scipy.stats import spearmanr
import c2h4_blind_score as S
key = S.load_key(S.C2H4 / "key" / "overrides_0b56238.tsv")
L = S.load_lines("unedited"); ids = np.array(L["id"]); src = np.array([i.split(".")[0] for i in ids])
main = L["in_main"] == 1
inN = np.isin(ids, list(key["negate"])); inS = np.isin(ids, list(S.source_negations())); inX = np.isin(src, list(key["exclude"]))
for nm in ["unedited", "edited"]:
    lv = np.array([float(r.split("\t")[2]) for r in (S.PRED / f"{nm}_levels.tsv").read_text().splitlines()[1:]])
    print(nm, "level sigma cm-1: min %.2e p5 %.2e median %.2e p95 %.2e max %.2e" % (lv.min(), *np.percentile(lv, [5, 50, 95]), lv.max()))
elig = main & ~inN & ~inS & ~inX
has = np.isin(ids, list(key["set_unc"])); scaled = np.isin(src, list(key["scale"]))
pool = (elig & has) | (elig & ~has & ~scaled); lab = has[pool]
inv_unc = 1.0 / L["unc"][pool]
print("POST HOC AUROC set_unc: 1/unc alone %.3f ; gnn_sigma_over_unc %.3f ; spearman(gnn_sigma_over_unc, 1/unc) over pool %.3f"
      % (S.auroc(lab, inv_unc), S.auroc(lab, L["gnn_sigma_over_unc"][pool]), spearmanr(L["gnn_sigma_over_unc"][pool], inv_unc).statistic))
ci = S.bootstrap(lab, {"g": L["gnn_sigma_over_unc"][pool], "u": inv_unc}, S.auroc, [("g", "u")])
print("  95% CI gnn_sigma_over_unc - 1/unc", ci["g-u"])
# outlier: spearman between gnn logit and resid/unc
print("POST HOC spearman(gnn logit, resid/unc) on main:", round(spearmanr(L["gnn_outlier_logit"][main], L["resid_over_unc"][main]).statistic, 3))
# edited: trivial baseline for states unc
st = {r.split("\t")[0]: (float(r.split("\t")[2]), int(r.split("\t")[3])) for r in (S.C2H4 / "C2H4_12C2H4_states.txt").read_text().splitlines()}
E = S.load_lines("edited"); em = E["in_main"] == 1
minunc = {}
for u_, l_, un, m in zip(E["upper"], E["lower"], E["unc"], em):
    if m:
        for a in (u_, l_): minunc[a] = min(minunc.get(a, 9), un)
lv = [r.split("\t") for r in (S.PRED / "edited_levels.tsv").read_text().splitlines()[1:]]
a = [(float(x[2]), st[x[0]][0], minunc[x[0]], st[x[0]][1]) for x in lv if st[x[0]][0] > 0]
g, u, mu, d = map(np.array, zip(*a))
print("POST HOC edited Spearman with states unc: gnn %.3f ; min incident line unc %.3f ; 1/degree %.3f"
      % (spearmanr(g, u).statistic, spearmanr(mu, u).statistic, spearmanr(1 / d, u).statistic))
