"""db_MARVEL headroom study: post hoc look at the headroom lines H (not in the
pre-registration). Are they visible to residuals at all, and do they cluster
by source? Run after scripts/dbmarvel_headroom.py (reads its score cache).

Run: uv run python scripts/dbmarvel_posthoc.py
"""

import json

import numpy as np

from dbmarvel_headroom import OUT, cached, records


def main():
    rows = {}
    for rec in records():
        s = cached(rec)
        idx = s["idx"]
        neg, x = rec["neg"][idx], rec["x"][idx]
        n = neg & ~x
        if n.sum() < 10 or "outlier" not in rec["usable"]:
            continue
        tags = np.array([rec["trans"][i].ref.rsplit(".", 1)[0] for i in idx])
        negfrac = {t: neg[tags == t].mean() for t in set(tags)}
        src_neg = np.array([negfrac[t] for t in tags])
        bridge = s["h"] >= 0.99
        order = s["order"]
        for tag, k in (("iv", s["k"]), ("iv2", s["k2"])):
            caught = (order >= 0) & (order < k)
            h = n & ~bridge & ~caught & ~s["rule"]
            neg_pool = ~neg
            rows.setdefault(rec["name"], {})[tag] = {
                "H": int(h.sum()),
                "H_z<3": float((s["z"][h] < 3).mean()) if h.any() else None,
                "H_stud<3": float((s["stud"][h] < 3).mean()) if h.any() else None,
                "neg_z<3": float((s["z"][neg_pool] < 3).mean()),
                "H_src_neg>=0.5": float((src_neg[h] >= 0.5).mean()) if h.any() else None,
                "H_median_z": float(np.median(s["z"][h])) if h.any() else None,
            }
    (OUT / "posthoc.json").write_text(json.dumps(rows, indent=1))
    for name, r in rows.items():
        a, b = r["iv"], r["iv2"]
        print(f"{name[:34]:34s} H {a['H']:5d} z<3 {a['H_z<3'] or 0:.2f} stud<3 {a['H_stud<3'] or 0:.2f} "
              f"(negatives z<3 {a['neg_z<3']:.2f}) src>=50%neg {a['H_src_neg>=0.5'] or 0:.2f} | "
              f"iv' H {b['H']:5d} z<3 {b['H_z<3'] or 0:.2f}")


if __name__ == "__main__":
    main()
