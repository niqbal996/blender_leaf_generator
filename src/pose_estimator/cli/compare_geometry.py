"""Re-score existing P3 geometry experiments without re-running models."""

import argparse
import json
from pathlib import Path
from typing import Optional

from pose_estimator.geometry import BACKENDS, compare_backends, sparse_model_dir


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS,
                        help="Existing experiments to compare; default discovers them")
    args = parser.parse_args(argv)
    backends = args.backends or [name for name in BACKENDS if sparse_model_dir(args.workdir, name).is_dir()]
    report = compare_backends(args.workdir, backends)
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
