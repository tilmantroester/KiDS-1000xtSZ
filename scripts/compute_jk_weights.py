import argparse
import numpy as np

import healpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-paths", nargs="+", required=True)
    parser.add_argument("--jk-def", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    jk_def = np.load(args.jk_def)

    print("Loading ", args.weight_paths[0])
    m = healpy.read_map(args.weight_paths[0], dtype=np.float64)
    for p in args.weight_paths[1:]:
        print("Loading ", p)
        m *= healpy.read_map(p, dtype=np.float64)

    jk_resolutions = [k for k in jk_def if k != "nside"]

    jk_weights = {}
    for jk_res in jk_resolutions:
        jk_weights[jk_res] = np.zeros(jk_def[jk_res].shape[0], dtype=np.float64)
        for i, idx in enumerate(jk_def[jk_res]):
            w = m[idx]
            w = w[w != healpy.UNSEEN]
            jk_weights[jk_res][i] = np.sum(w)

    np.savez(args.output, **jk_weights)