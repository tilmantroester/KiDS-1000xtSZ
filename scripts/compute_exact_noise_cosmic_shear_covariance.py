import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import printflush
from compute_cosmic_shear_covariance import compute_gaussian_covariance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pymaster-workspace-path", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument("--idx-a")
    parser.add_argument("--idx-b")

    args = parser.parse_args()

    idx_a1, idx_a2 = [int(s) for s in args.idx_a.split("-")]
    idx_b1, idx_b2 = [int(s) for s in args.idx_b.split("-")]

    print(f"Computing exact noise cosmic shear covariance matrices for "
          f"{idx_a1}-{idx_a2}, {idx_b1}-{idx_b2}")

    if not ((idx_a1 == idx_b1 and idx_a2 == idx_b2)
            or (idx_a1 == idx_b2 and idx_a2 == idx_b1)):
        print("Nothing to do for pure noise covariance.")
        sys.exit(0)

    workspace_path = args.pymaster_workspace_path
    printflush("Reading Cl workspaces from ", workspace_path)
    wsp_a = nmt.NmtWorkspace()
    wsp_a.read_from(os.path.join(workspace_path,
                    f"pymaster_workspace_shear_{idx_a1}_shear_{idx_a2}.fits"))
    wsp_b = nmt.NmtWorkspace()
    wsp_b.read_from(os.path.join(workspace_path,
                    f"pymaster_workspace_shear_{idx_b1}_shear_{idx_b2}.fits"))

    printflush("Reading covariance workspaces")
    cov_wsp = {}
    cov_wsp["nnnn"] = nmt.NmtCovarianceWorkspace()
    cov_wsp["nnnn"].read_from(
                        os.path.join(
                            workspace_path,
                            f"pymaster_cov_workspace_"
                            f"shear_noise_{idx_a1}_shear_noise_{idx_a2}_"
                            f"shear_noise_{idx_b1}_shear_noise_{idx_b2}.fits"))

    ell = np.arange(3*2048)

    os.makedirs(args.output_path, exist_ok=True)
    cov_filename = (f"cov_shear_noise_{idx_a1}_shear_noise_{idx_a2}_"
                    f"shear_noise_{idx_b1}_shear_noise_{idx_b2}.npz")
    cov_filename = os.path.join(args.output_path, cov_filename)
    print("Saving covariance matrices to ", cov_filename)

    printflush("Computing covariance matrices")
    cov_matrix = {}

    exact_noise_terms = [idx_a1 == idx_b1,
                         idx_a1 == idx_b2,
                         idx_a2 == idx_b1,
                         idx_a2 == idx_b2]

    cov_matrix["nnnn"] = compute_gaussian_covariance(
                                idx_a1, idx_a2, idx_b1, idx_b2,
                                signal_terms=[False, False, False, False],
                                noise_terms=[False, False, False, False],
                                exact_noise_terms=exact_noise_terms,
                                Cl_signal=None,
                                Cl_noise=None,
                                ell=ell,
                                cov_wsp=cov_wsp["nnnn"],
                                wsp_a=wsp_a, wsp_b=wsp_b)
    cov_matrix["nnnn"] *= healpy.nside2pixarea(2048)**2

    np.savez(cov_filename, **cov_matrix)
