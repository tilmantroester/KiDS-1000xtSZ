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

    Cls = []
    for idx, is_signal, is_noise, is_exact_noise in zip(
                   [(idx_a1, idx_b1),
                    (idx_a1, idx_b2),
                    (idx_a2, idx_b1),
                    (idx_a2, idx_b2)],
                   signal_terms, noise_terms, exact_noise_terms):
        if is_exact_noise:
            Cl = np.array([Cl_1, Cl_0, Cl_0, Cl_1])
        else:
            Cl = np.array([Cl_0, Cl_0, Cl_0, Cl_0])
            if is_signal:
                Cl += Cl_signal[idx]
            if is_noise:
                Cl += Cl_noise[idx]
        Cls.append(Cl)

    # print("Cls for nmt.gaussian_covariance")
    # print(Cls)
    cov = nmt.gaussian_covariance(cov_wsp,
                                  spin_a1=2, spin_a2=2, spin_b1=2, spin_b2=2,
                                  cla1b1=Cls[0],
                                  cla1b2=Cls[1],
                                  cla2b1=Cls[2],
                                  cla2b2=Cls[3],
                                  wa=wsp_a, wb=wsp_b)

    return cov


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--Cl-cov-file", required=True)
    parser.add_argument("--pymaster-workspace-path", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument("--idx-a")
    parser.add_argument("--idx-b")

    parser.add_argument("--compute-nka-terms", action="store_true")
    parser.add_argument("--compute-exact-noise-mixed-terms",
                        action="store_true")

    args = parser.parse_args()

    idx_a1, idx_a2 = [int(s) for s in args.idx_a.split("-")]
    idx_b1, idx_b2 = [int(s) for s in args.idx_b.split("-")]

    use_nka = args.compute_nka_terms
    use_exact_noise_mixed_terms = args.compute_exact_noise_mixed_terms

    print(f"Computing cosmic shear covariance matrices for "
          f"{idx_a1}-{idx_a2}, {idx_b1}-{idx_b2}")
    if use_nka:
        print("Computing NKA terms.")
    if use_exact_noise_mixed_terms:
        print("Computing exact noise mixed terms.")

    if not use_nka and use_exact_noise_mixed_terms:
        if (idx_a1 != idx_b1 and idx_a1 != idx_b2
                and idx_a2 != idx_b1 and idx_a2 != idx_b2):
            printflush("Nothing to do for exact mixed terms.")
            sys.exit(0)

    os.makedirs(args.output_path, exist_ok=True)

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

    cov_wsp = {}
    cov_matrix = {}

    cov_filename = (f"cov_shear_{idx_a1}_shear_{idx_a2}_"
                    f"shear_{idx_b1}_shear_{idx_b2}.npz")
    cov_filename = os.path.join(args.output_path, cov_filename)
    printflush("Saving covariance matrices to ", cov_filename)

    if use_nka:
        exact_noise_terms = [False, False, False, False]

        wsp_file = os.path.join(workspace_path,
                                f"pymaster_cov_workspace_"
                                f"shear_{idx_a1}_shear_{idx_a2}_"
                                f"shear_{idx_b1}_shear_{idx_b2}.fits")
        printflush(f"Reading covariance workspace {wsp_file}")
        cov_wsp = nmt.NmtCovarianceWorkspace()
        cov_wsp.read_from(wsp_file)

        printflush("Computing covariance matrices")
        cov_matrix["aaaa"] = compute_gaussian_covariance(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    signal_terms=[True, True, True, True],
                                    noise_terms=[True, True, True, True],
                                    exact_noise_terms=exact_noise_terms,
                                    Cl_signal=Cl_signal,
                                    Cl_noise=Cl_noise,
                                    ell=ell,
                                    cov_wsp=cov_wsp,
                                    wsp_a=wsp_a, wsp_b=wsp_b)

        cov_matrix["ssss"] = compute_gaussian_covariance(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    signal_terms=[True, True, True, True],
                                    noise_terms=[False, False, False, False],
                                    exact_noise_terms=exact_noise_terms,
                                    Cl_signal=Cl_signal,
                                    Cl_noise=Cl_noise,
                                    ell=ell,
                                    cov_wsp=cov_wsp,
                                    wsp_a=wsp_a, wsp_b=wsp_b)

        cov_matrix["nnnn"] = compute_gaussian_covariance(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    signal_terms=[False, False, False, False],
                                    noise_terms=[True, True, True, True],
                                    exact_noise_terms=exact_noise_terms,
                                    Cl_signal=Cl_signal,
                                    Cl_noise=Cl_noise,
                                    ell=ell,
                                    cov_wsp=cov_wsp,
                                    wsp_a=wsp_a, wsp_b=wsp_b)

    if use_exact_noise_mixed_terms:
        noise_fields = []
        exact_noise_terms = []
        signal_terms = []

        if idx_a1 == idx_b1:
            noise_fields.append([True, False, True, False])
            exact_noise_terms.append([True, False, False, False])
            signal_terms.append([False, False, False, True])
        if idx_a1 == idx_b2:
            noise_fields.append([True, False, False, True])
            exact_noise_terms.append([False, True, False, False])
            signal_terms.append([False, False, True, False])
        if idx_a2 == idx_b1:
            noise_fields.append([False, True, True, False])
            exact_noise_terms.append([False, False, True, False])
            signal_terms.append([False, True, False, False])
        if idx_a2 == idx_b2:
            noise_fields.append([False, True, False, True])
            exact_noise_terms.append([False, False, False, True])
            signal_terms.append([True, False, False, False])

        tags_mixed = []

        for n_field, n_term, s_term in zip(noise_fields,
                                           exact_noise_terms,
                                           signal_terms):
            wsp_tag = "_".join([("shear_noise" if n else "shear") + f"_{i}"
                                for n, i in zip(n_field,
                                                [idx_a1, idx_a2,
                                                 idx_b1, idx_b2])])
            wsp_file = os.path.join(workspace_path,
                                    f"pymaster_cov_workspace_{wsp_tag}.fits")
            printflush(f"Reading covariance workspace {wsp_file}")
            cov_wsp = nmt.NmtCovarianceWorkspace()
            cov_wsp.read_from(wsp_file)

            tag = "".join(["n" if n else "s" for n in n_field])
            tags_mixed.append(tag)

            printflush("Computing covariance matrix ", tag)

            cov_matrix[tag] = compute_gaussian_covariance(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    signal_terms=s_term,
                                    noise_terms=[False, False, False, False],
                                    exact_noise_terms=n_term,
                                    Cl_signal=Cl_signal,
                                    Cl_noise=None,
                                    ell=ell,
                                    cov_wsp=cov_wsp,
                                    wsp_a=wsp_a, wsp_b=wsp_b)
            cov_matrix[tag] *= healpy.nside2pixarea(2048)

        cov_matrix["mmmm"] = sum([cov_matrix[t] for t in tags_mixed])

    np.savez(cov_filename, **cov_matrix)
