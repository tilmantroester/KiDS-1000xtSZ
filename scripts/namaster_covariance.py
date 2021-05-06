import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--shear-w-maps", nargs="+")
    parser.add_argument("--shear-w2e2-maps", nargs="+")

    parser.add_argument("--pymaster-workspace-output-path")

    parser.add_argument("--map-names", nargs="+")

    parser.add_argument("--n-iter")

    args = parser.parse_args()

    n_iter = 3
    if args.n_iter is not None:
        n_iter = int(args.n_iter)
        print(f"Using n_iter = {n_iter}")

    pymaster_workspace_output_path = args.pymaster_workspace_output_path
    os.makedirs(pymaster_workspace_output_path, exist_ok=True)

    w_fields = []
    w_sigma_fields = []
    print("Loading maps")
    for weight_map_file, w2e2_map_file in zip(args.shear_w_maps,
                                              args.shear_w2e2_maps):
        # print(weight_map_file)
        # w = healpy.read_map(weight_map_file, verbose=False)
        # w[w == healpy.UNSEEN] = 0

        print(w2e2_map_file)
        w2e2 = healpy.read_map(w2e2_map_file, verbose=False)
        w2e2[w2e2 == healpy.UNSEEN] = 0

        nside = healpy.get_nside(w2e2)

        print("  Creating field objects")
        # w_field = nmt.NmtField(w, [None], n_iter=n_iter)
        w_sigma_field = nmt.NmtField(np.sqrt(w2e2), None, n_iter=n_iter, spin=0)

        # w_fields.append(w_field)
        w_sigma_fields.append(w_sigma_field)

    field_idx = [(i, j) for i in range(len(w_sigma_fields))
                 for j in range(i+1)]

    for i, (idx_a1, idx_a2) in enumerate(field_idx):
        print(f"  A {idx_a1}-{idx_a2}")
        for idx_b1, idx_b2 in field_idx[:i+1]:
            print(f"    B {idx_b1}-{idx_b2}")

            if ((idx_a1 == idx_b1 and idx_a2 == idx_b2)
                    or (idx_a1 == idx_b2 and idx_a2 == idx_b1)):
                print("Computing noise-noise covariance")
                nmt_cov_workspace = nmt.NmtCovarianceWorkspace()

                nmt_cov_workspace.compute_coupling_coefficients(
                                        fla1=w_sigma_fields[idx_a1],
                                        fla2=w_sigma_fields[idx_a2],
                                        flb1=w_sigma_fields[idx_b1],
                                        flb2=w_sigma_fields[idx_b2])

                nmt_cov_workspace.write_to(
                    os.path.join(
                        pymaster_workspace_output_path,
                        f"pymaster_cov_workspace"
                        f"_shear_noise_{idx_a1}_shear_noise_{idx_a2}"
                        f"_shear_noise_{idx_b1}_shear_noise_{idx_b2}.fits"))
