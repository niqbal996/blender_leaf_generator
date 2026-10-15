"""Which P5x leaves belong to an apex, by apex_select's rule and by apex_fit's, side by side.

    PYTHONPATH=src:scripts python scripts/apex_membership.py <workdir> [<workdir> ...]

apex_select (until 2026-10-14) replaced a P5x leaf when its curve's middle point lay within the
region's reach (sphere round the region's centre, 95th-percentile radius); apex_fit drops a P5x leaf (the fit
replaces it) when more than half of its points lie in the carved region, and renders the rest
as context. A leaf the first rule takes and the second does not is neither fitted nor kept:
it was missing from the fit's skeleton. apex_select now uses apex_fit's rule for P5x leaves
(replaced_by_apex); this prints what the old rule would have lost. Reads <workdir>/apex/regions/apex_regions.npz.
"""
import sys
from pathlib import Path

import numpy as np

from apex_select import in_region, leaf_world, region_reach, rows_in
from p5x_views import Run


def main():
    for wd in map(Path, sys.argv[1:]):
        run = Run(wd)
        regions = dict(np.load(wd / "apex" / "regions" / "apex_regions.npz"))
        reach = region_reach(regions)
        lab = run.assignment
        leaves = run.skeleton["leaves"]
        print(f"{wd.parent.name}: {len(leaves)} P5x leaves")
        lost_all = set()
        for name, pts in sorted(regions.items()):
            inside = rows_in(run.points, pts)
            by_old = {int(l["id"]) for l in leaves if in_region(leaf_world(run, l), reach[name])}
            by_fit = {int(l["id"]) for l in leaves if (lab == l["id"]).any() and inside[lab == l["id"]].mean() > 0.5}
            lost = by_old - by_fit
            lost_all |= lost
            print(f"  {name}: old select rule: {sorted(by_old)}, fit replaces {sorted(by_fit)}; "
                  f"neither fitted nor kept: {sorted(lost)}; fitted over but kept too: {sorted(by_fit - by_old)}")
        print(f"  the old rule lost: {len(lost_all)} of {len(leaves)}")


if __name__ == "__main__":
    main()
