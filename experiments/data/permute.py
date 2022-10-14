# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import jsonlines
import numpy as np
from pathlib import Path


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=3435)
    parser.add_argument("--percentage", type=float, default=100)
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    lines = [line for line in jsonlines.open(Path(args.in_file), "r")]
    rng = np.random.RandomState(args.seed)
    perm_idxs = rng.permutation(len(lines))
    perm_lines = [lines[i] for i in perm_idxs]

    N = len(perm_lines)
    n = round((args.percentage / 100.0) * N)
    print(f"Use {args.percentage}%: {N} => {n}")

    with jsonlines.open(Path(args.out_file), "w") as out:
        out.write_all(perm_lines[:n])


if __name__ == "__main__":
    main()
