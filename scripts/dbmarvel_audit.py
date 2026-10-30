"""db_MARVEL headroom study, phase 1: label audit.

Run: uv run python scripts/dbmarvel_audit.py

Reads the read-only copy in data/dbmarvel/ (db_MARVEL 010b888, see
PROVENANCE.txt / SHA256SUMS): manifests.json (the copied YAML manifests,
resolved by db_MARVEL's own loader/manifest.py) and the raw run files, parsed
with db_MARVEL's loader/parse.py. Plus 12C2H4 v1 (data/C2H4_v1, MARVEL_network
0b56238) as dataset 22. Writes reports/2026-10-30_dbmarvel_headroom/audit.tsv.

Per run: levels and lines of the unedited graph (every line, negations
flipped back in), largest component, negations, original vs used
uncertainty, notes, source units (manifest vs inferred from the published
levels), and an edited-graph re-solve against the published levels.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

from marvel_gnn.core.parse import UNC_FLOOR, Transition
from marvel_gnn.core.solver import solve_energies

ROOT = Path(__file__).parents[1]
DB = ROOT / "data" / "dbmarvel"
OUT = ROOT / "reports" / "2026-10-30_dbmarvel_headroom"
sys.path.insert(0, str(DB / "loader"))
from parse import parse_levels, parse_transitions  # noqa: E402  (db_MARVEL's parser)

# 24TeFuYu (N2O 446) has no manifest in db_MARVEL; layout read off the files:
# freq og_unc used_unc, 5 QN tokens per level (names unverified), no segment
# file, so every source's unit is inferred.
EXTRA = {"24TeFuYu": {"publication": {}, "sources": [], "datasets": [{
    "formula": "N2O", "isotopologue": "14N2-16O", "marvel_version": "unknown",
    "completeness": "complete", "n_unc_cols": 2, "qn_names": ["q1", "q2", "q3", "ef", "J"],
    "level_columns": ["q1", "q2", "q3", "ef", "J", "energy", "uncertainty", "n_transitions"],
    "files": {"input_transitions": "n2o_446_transitions.txt",
              "output_levels": "n2o_446_levels.txt"}}]}}

# cm-1 per unit
UNIT = {"cm-1": 1.0, "GHz": 1 / 29.9792458, "MHz": 1 / 29979.2458, "kHz": 1 / 29979245.8}


def runs():
    """(publication, dataset dict, {source tag: unit}) for every run with transitions."""
    man = json.loads((DB / "manifests.json").read_text()) | EXTRA
    for pub, m in man.items():
        units = {s["tag"]: s["unit"] for s in m["sources"]}
        for d in m["datasets"]:
            if "input_transitions" in d["files"]:
                yield pub, d, units


def parsed(pub, d):
    """db_MARVEL ParsedTransitions. numeric_field_widths (20YiOwTe) mis-slices
    rows whose unc field runs past 12 chars (675 rows: unc lost, upper label
    clipped 'muA2Sigma' -> 'uA2Sigma'), so instead split only the 206 glued
    rows (first token holds two '.') at the declared freq width."""
    path = DB / "data" / pub / d["files"]["input_transitions"]
    widths = d.get("numeric_field_widths")
    if widths:
        fixed = DB / "fixed" / f"{pub}_{path.name}"
        fixed.parent.mkdir(exist_ok=True)
        rows = [r[:widths[0]] + " " + r[widths[0]:] if r.split() and r.split()[0].count(".") > 1
                else r for r in path.read_text().splitlines()]
        fixed.write_text("\n".join(rows) + "\n")
        path = fixed
    return parse_transitions(path, d["qn_names"], d["n_unc_cols"])


def published_levels(pub, d):
    """{assignment: (energy, unc)} from the run's output levels file."""
    return {" ".join(lv.qn): (lv.energy, lv.uncertainty) for lv in
            parse_levels(DB / "data" / pub / d["files"]["output_levels"],
                         d["level_columns"], d["qn_names"])}


def infer_units(tr, levels):
    """{source: unit} from median |freq| / (E_u - E_l) over lines whose levels
    are both published, snapped to the nearest UNIT in log space."""
    ratio = {}
    for t in tr:
        u, l = levels.get(" ".join(t.upper_qn)), levels.get(" ".join(t.lower_qn))
        if u and l and u[0] - l[0] > 1e-9:
            ratio.setdefault(t.source_tag, []).append(abs(t.obs_freq) / (u[0] - l[0]))
    names = list(UNIT)
    logs = np.log10([1 / UNIT[n] for n in names])
    return {s: names[int(np.argmin(np.abs(logs - np.log10(np.median(r)))))]
            for s, r in ratio.items()}


def load(pub, d, units, levels=None):
    """Unedited graph of one run -> (transitions, negated bool array).

    Transition.unc = used unc, .orig_unc = original unc (equal for 1-column
    files), both in cm-1; freq = |obs_freq| in cm-1. Units from the manifest,
    else inferred from the published levels. Missing unc: the other column,
    else UNC_FLOOR. ref = 'TAG.n'.
    """
    tr = parsed(pub, d)
    units = units | {s: u for s, u in infer_units(tr, levels or published_levels(pub, d)).items()
                     if s not in units}
    out = []
    for t in tr:
        c = UNIT[units.get(t.source_tag, "cm-1")]
        og = t.og_unc_freq or t.used_unc_freq or UNC_FLOOR
        used = t.used_unc_freq or og
        out.append(Transition(abs(t.obs_freq) * c, used * c, og * c, " ".join(t.upper_qn),
                              " ".join(t.lower_qn), f"{t.source_tag}.{t.source_number}"))
    return out, np.array([t.obs_freq < 0 for t in tr])


def main_component(trans):
    g = nx.Graph((t.upper, t.lower) for t in trans)
    main = max(nx.connected_components(g), key=len)
    return [k for k, t in enumerate(trans) if t.upper in main]


def audit(pub, d, units):
    row = {"publication": pub, "molecule": d["formula"], "isotopologue": d["isotopologue"],
           "version": d["marvel_version"], "n_unc_cols": d["n_unc_cols"],
           "qn": " ".join(d["qn_names"]), "problems": []}
    tr = parsed(pub, d)
    levels = published_levels(pub, d)
    trans, neg = load(pub, d, units, levels)
    g = nx.Graph((t.upper, t.lower) for t in trans)
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    main = comps[0]
    in_main = np.array([t.upper in main for t in trans])
    row |= {"lines": len(trans), "levels": g.number_of_nodes(), "n_comp": len(comps),
            "largest": len(main), "lines_largest": int(in_main.sum()),
            "out_levels": len(levels), "negated": int(neg.sum()),
            "neg_in_largest": int((neg & in_main).sum())}
    tags = Counter(t.source_tag for t in tr)
    negtags = Counter(t.source_tag for t in tr if t.obs_freq < 0)
    whole = [s for s in negtags if negtags[s] == tags[s]]
    row["neg_whole_source"] = f"{sum(tags[s] for s in whole)} ({len(whole)} src)"
    in_whole = np.array([t.source_tag in whole for t in tr])
    row["N_main"] = int((neg & in_main & ~in_whole).sum())
    row["self_loops"] = sum(t.upper == t.lower for t in trans)
    row["unc_missing"] = sum(not (t.og_unc_freq and t.used_unc_freq) for t in tr)
    if d["n_unc_cols"] == 2:
        og = np.array([t.orig_unc for t in trans])
        used = np.array([t.unc for t in trans])
        row["unc_raised"] = int(((used > og * (1 + 1e-9)) & ~neg).sum())
        row["unc_lowered"] = int(((used < og * (1 - 1e-9)) & ~neg).sum())
    notes = Counter(t.note for t in tr if t.note)
    row["noted"] = sum(notes.values())
    row["notes_top"] = "; ".join(f"{k}:{v}" for k, v in notes.most_common(3))
    row["sources"] = len(tags)
    inferred = infer_units(tr, levels)
    row["non_cm1_lines"] = sum(n for s, n in tags.items()
                               if units.get(s, inferred.get(s, "cm-1")) != "cm-1")
    row["unit_mismatch"] = " ".join(f"{s}:{units[s]}->{u}" for s, u in inferred.items()
                                    if s in units and units[s] != u)
    row["unit_inferred"] = sum(s not in units for s in tags)
    dup = Counter((t.source_tag, t.source_number) for t in tr)
    row["dup_tags"] = sum(v > 1 for v in dup.values())

    # edited graph (negated lines dropped, used unc), largest component, re-solved
    ed = [t for t, n in zip(trans, neg) if not n]
    ed = [ed[k] for k in main_component(ed)]
    e = solve_energies(ed)
    r = np.array([abs(t.freq - (e[t.upper] - e[t.lower])) for t in ed])
    used = np.array([t.unc for t in ed])
    og = np.array([t.orig_unc for t in ed])
    row["ed_inconsistent"] = int((r > used * 1.001).sum())
    raised = used > og * (1 + 1e-9)
    if raised.any():  # MARVEL 4-style automatic adjustment: used unc ~ final |resid|
        row["raised_at_resid"] = int((raised & (r > 0.9 * used)).sum())
    common = [a for a in e if a in levels]
    off = np.median([e[a] - levels[a][0] for a in common]) if common else np.nan
    dev = np.array([abs(e[a] - off - levels[a][0]) / max(levels[a][1], 1e-6) for a in common])
    row["pub_match"] = f"{len(common)}/{len(e)}"
    row["pub_within_unc"] = f"{np.mean(dev <= 1):.3f}" if common else ""
    if d["n_unc_cols"] == 2:  # raised among the N_main-pool lines, at the edited re-solve
        row["U_main"] = int(raised.sum())
    row["usable"] = usable(row)
    if row["unc_missing"]:
        row["problems"].append(f"{row['unc_missing']} lines with blank/0 unc")
    return row


MIN_POS = 10  # positives needed in the largest component to score a run


def usable(row):
    """outlier: >= MIN_POS network negations (N_main); uncertainty: original and
    used unc both present with >= MIN_POS raised lines in the edited largest
    component, and published levels reproduced (pub_within_unc >= 0.9)."""
    ok = float(row.get("pub_within_unc") or 0) >= 0.9
    out = [k for k, n in (("outlier", row["N_main"]), ("uncertainty", row.get("U_main", 0)))
           if n >= MIN_POS and ok]
    return "+".join(out) or "no"


def audit_c2h4():
    """12C2H4 v1: unedited lines from the 0b56238 inputs; labels from the key."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from c2h4_blind_predict import load_unedited
    lines, flipped = load_unedited()
    key = [r.split("\t") for r in
           (ROOT / "data/C2H4_v1/key/overrides_0b56238.tsv").read_text().splitlines()[1:] if r]
    acts = Counter(r[0] for r in key)
    g = nx.Graph((t.upper, t.lower) for t in lines)
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    return {"publication": "MARVEL_network@0b56238", "molecule": "C2H4", "isotopologue": "12C2H4",
            "version": "4 (v1)", "n_unc_cols": "key", "qn": "v1..v12 J Ka Kc",
            "lines": len(lines), "levels": g.number_of_nodes(), "n_comp": len(comps),
            "largest": len(comps[0]), "negated": acts["negate"] + len(flipped),
            "neg_whole_source": "54 (1 src, 58AlPl)", "N_main": 544 + 81,  # 91 source negations less 58AlPl's 10 (in X)
            "unc_raised": acts["set_unc"], "U_main": acts["set_unc"],
            "usable": "outlier+uncertainty (key)",
            "notes_top": f"key: {dict(acts)}; source negations {len(flipped)}",
            "problems": ["6 sources blank unc (policy-filled)"]}


COLS = ["publication", "usable", "molecule", "isotopologue", "version", "n_unc_cols", "qn", "lines",
        "levels", "n_comp", "largest", "lines_largest", "out_levels", "pub_match",
        "pub_within_unc", "negated", "neg_in_largest", "neg_whole_source", "N_main", "unc_raised", "U_main",
        "raised_at_resid", "unc_lowered", "ed_inconsistent", "unc_missing", "self_loops",
        "dup_tags", "sources", "non_cm1_lines", "unit_inferred", "unit_mismatch", "noted",
        "notes_top", "problems"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in runs():
        try:
            rows.append(audit(*r))
        except Exception as e:  # report, don't stop the audit
            rows.append({"publication": r[0], "isotopologue": r[1]["isotopologue"],
                         "problems": [f"{type(e).__name__}: {e}"[:300]]})
    rows.append(audit_c2h4())
    text = ["\t".join(COLS)] + ["\t".join(" | ".join(r.get(c, [])) if c == "problems"
                                          else str(r.get(c, "")) for c in COLS) for r in rows]
    (OUT / "audit.tsv").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
