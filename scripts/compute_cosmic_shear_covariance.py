import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import printflush


def compute_gaussian_covariance(idx_a1, idx_a2, idx_b1, idx_b2,
                                signal_terms, noise_terms, exact_noise_terms,
                                Cl_signal, Cl_noise, ell,
                                cov_wsp, wsp_a, wsp_b):
    Cl_0 = np.zeros(len(ell))
    Cl_1 = np.ones(len(ell))

    Cls = {}
    for idx, is_signal, is_noise, is_exact_noise in zip(
                   [(idx_a1, idx_b1),
                    (idx_a1, idx_b2),
                    (idx_a2, idx_b1),
                    (idx_a2, idx_b2)],
                   signal_terms, noise_terms, exact_noise_terms):
        Cls[idx] = np.zeros_like(ell)
        if is_exact_noise:
            Cls[idx] = [Cl_1, Cl_0, Cl_0, Cl_1]
        else:
            Cls[idx] = [Cl_0, Cl_0, Cl_0, Cl_0]
            if is_signal:
                Cls[idx] += Cl_signal[idx]
            if is_noise:
                Cls[idx] += Cl_noise[idx]

    cov = nmt.gaussian_covariance(cov_wsp,
                                  spin_a1=2, spin_a2=2, spin_b1=2, spin_b2=2,
                                  cla1b1=Cls[(idx_a1, idx_b1)],
                                  cla1b2=Cls[(idx_a1, idx_b2)],
                                  cla2b1=Cls[(idx_a2, idx_b1)],
                                  cla2b2=Cls[(idx_a2, idx_b2)],
                                  wa=wsp_a, wb=wsp_b)

    return cov


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--Cl-cov-file", required=True)
    parser.add_argument("--pymaster-workspace-path", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument("--idx-a")
    parser.add_argument("--idx-b")

    args = parser.parse_args()

    idx_a1, idx_a2 = [int(s) for s in args.idx_a.split("-")]
    idx_b1, idx_b2 = [int(s) for s in args.idx_b.split("-")]

    print(f"Computing cosmic shear covariance matrices for "
          f"{idx_a1}-{idx_a2}, {idx_b1}-{idx_b2}")

    Cl_signal = {}
    Cl_noise = {}
    Cl_filename_template = args.Cl_cov_file + "{}-{}.npz"
    printflush("Reading Cls from ", Cl_filename_template)
    for idx in [(idx_a1, idx_b1),
                (idx_a1, idx_b2),
                (idx_a2, idx_b1),
                (idx_a2, idx_b2)]:
        Cl_filename = Cl_filename_template.format(*idx)
        if os.path.isfile(Cl_filename):
            d = np.load(Cl_filename)
        else:
            Cl_filename = Cl_filename_template.format(*idx[::-1])
            if os.path.isfile(Cl_filename):
                d = np.load(Cl_filename)
            else:
                raise ValueError(f"Cannot find covariance Cl file for {idx}: "
                                 f"{Cl_filename}")
        Cl_signal[idx] = d["Cl_cov"]
        Cl_noise[idx] = d["Cl_noise_cov"]
        ell = d["ell"]

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
    cov_wsp["ssss"] = nmt.NmtCovarianceWorkspace()
    cov_wsp["ssss"].read_from(os.path.join(
                              workspace_path,
                              f"pymaster_cov_workspace_"
                              f"shear_{idx_a1}_shear_{idx_a2}_"
                              f"shear_{idx_b1}_shear_{idx_b2}.fits"))

    printflush("Computing covariance matrices")
    cov_matrix = {}
    cov_matrix["ssss"] = compute_gaussian_covariance(
                                idx_a1, idx_a2, idx_b1, idx_b2,
                                signal_terms=[True, True, True, True],
                                noise_terms=[True, True, True, True],
                                exact_noise_terms=[False, False, False, False],
                                Cl_signal=Cl_signal,
                                Cl_noise=Cl_noise,
                                ell=ell,
                                cov_wsp=cov_wsp["ssss"],
                                wsp_a=wsp_a, wsp_b=wsp_b)

    cov_filename = f"cov_shear_shear_{idx_a1}-{idx_a2}_{idx_b1}-{idx_b2}.npz"
    cov_filename = os.path.join(args.output_path, cov_filename)
    print("Saving covariance matrices to ", cov_filename)
    np.savez(cov_filename, **cov_matrix)
