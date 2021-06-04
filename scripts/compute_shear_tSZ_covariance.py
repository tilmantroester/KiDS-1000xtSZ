import argparse
import os

import pymaster as nmt

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import printflush  # noqa: E402


def compute_gaussian_covariance(idx_a1, idx_a2, idx_b1, idx_b2,
                                f_a1, f_a2, f_b1, f_b2,
                                spins,
                                Cl_signal, Cl_noise, ell,
                                cov_wsp, wsp_a, wsp_b):
    Cls = []
    for field, idx in [((f_a1, f_b1), (idx_a1, idx_b1)),
                       ((f_a1, f_b2), (idx_a1, idx_b2)),
                       ((f_a2, f_b1), (idx_a2, idx_b1)),
                       ((f_a2, f_b2), (idx_a2, idx_b2))]:
        Cls.append(Cl_signal[field][idx] + Cl_noise[field][idx])

    cov = nmt.gaussian_covariance(cov_wsp,
                                  spin_a1=spins[f_a1], spin_a2=spins[f_a2],
                                  spin_b1=spins[f_b1], spin_b2=spins[f_b2],
                                  cla1b1=Cls[0],
                                  cla1b2=Cls[1],
                                  cla2b1=Cls[2],
                                  cla2b2=Cls[3],
                                  wa=wsp_a, wb=wsp_b)

    return cov


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--Cl-cov-file", required=True, nargs="+")
    parser.add_argument("--pymaster-workspace-path-a", required=True)
    parser.add_argument("--pymaster-workspace-path-b", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument("--idx-a")
    parser.add_argument("--idx-b")

    parser.add_argument("--fields-a")
    parser.add_argument("--fields-b")

    args = parser.parse_args()

    idx_a1, idx_a2 = [int(s) for s in args.idx_a.split("-")]
    idx_b1, idx_b2 = [int(s) for s in args.idx_b.split("-")]

    f_a1, f_a2 = args.fields_a.split("-")
    f_b1, f_b2 = args.fields_b.split("-")

    spins = {"shear": 2, "foreground": 0}

    print(f"Computing cosmic shear covariance matrices for "
          f"{idx_a1}-{idx_a2}, {idx_b1}-{idx_b2} "
          f"({f_a1}-{f_a2}, {f_b1}-{f_b2})")

    Cl_signal = {}
    Cl_noise = {}

    if len(args.Cl_cov_file) % 2 != 0:
        raise ValueError("--Cl-cov-file requires "
                         "fields-file root pairs.")
    field_pairs = [f.split("-") for f in args.Cl_cov_file[::2]]
    file_templates = args.Cl_cov_file[1::2]

    Cl_filename_templates = {tuple(pair): fn
                             for pair, fn in zip(field_pairs, file_templates)}

    for field, idx in [((f_a1, f_b1), (idx_a1, idx_b1)),
                       ((f_a1, f_b2), (idx_a1, idx_b2)),
                       ((f_a2, f_b1), (idx_a2, idx_b1)),
                       ((f_a2, f_b2), (idx_a2, idx_b2))]:
        if field not in Cl_signal:
            Cl_signal[field] = {}
            Cl_noise[field] = {}

        try:
            Cl_filename = Cl_filename_templates[field].format(*idx)
            d = np.load(Cl_filename)
            printflush(f"Reading Cls for {idx[0]}-{idx[1]} "
                       f"({field[0]}-{field[1]})"
                       f" from ", Cl_filename)
        except (KeyError, OSError):
            Cl_filename = Cl_filename_templates[field[::-1]].format(*idx[::-1])
            printflush(f"Reading Cls for {idx[0]}-{idx[1]} "
                       f"({field[0]}-{field[1]})"
                       f" from ", Cl_filename)
            d = np.load(Cl_filename)

        Cl_signal[field][idx] = d["Cl_cov"]
        Cl_noise[field][idx] = d["Cl_noise_cov"]
        ell = d["ell"]

    workspace_path_a = args.pymaster_workspace_path_a
    workspace_path_b = args.pymaster_workspace_path_b
    printflush("Reading Cl workspace from ", workspace_path_a)
    wsp_a = nmt.NmtWorkspace()
    wsp_a.read_from(os.path.join(
                    workspace_path_a,
                    f"pymaster_workspace_"
                    f"{f_a1}_{idx_a1}_{f_a2}_{idx_a2}.fits"))
    printflush("Reading Cl workspace from ", workspace_path_b)
    wsp_b = nmt.NmtWorkspace()
    wsp_b.read_from(os.path.join(
                    workspace_path_b,
                    f"pymaster_workspace_"
                    f"{f_b1}_{idx_b1}_{f_b2}_{idx_b2}.fits"))

    workspace_file = os.path.join(workspace_path_a,
                                  f"pymaster_cov_workspace_"
                                  f"{f_a1}_{idx_a1}_{f_a2}_{idx_a2}_"
                                  f"{f_b1}_{idx_b1}_{f_b2}_{idx_b2}.fits")
    if not os.path.isfile(workspace_file):
        workspace_file = os.path.join(workspace_path_b,
                                      f"pymaster_cov_workspace_"
                                      f"{f_a1}_{idx_a1}_{f_a2}_{idx_a2}_"
                                      f"{f_b1}_{idx_b1}_{f_b2}_{idx_b2}.fits")
    printflush("Reading covariance workspace from ", workspace_file)
    cov_wsp = {}
    cov_wsp["ssss"] = nmt.NmtCovarianceWorkspace()
    cov_wsp["ssss"].read_from(workspace_file)

    printflush("Computing covariance matrices")
    cov_matrix = {}
    cov_matrix["ssss"] = compute_gaussian_covariance(
                                idx_a1, idx_a2, idx_b1, idx_b2,
                                f_a1, f_a2, f_b1, f_b2,
                                spins=spins,
                                Cl_signal=Cl_signal,
                                Cl_noise=Cl_noise,
                                ell=ell,
                                cov_wsp=cov_wsp["ssss"],
                                wsp_a=wsp_a, wsp_b=wsp_b)

    cov_filename = (f"cov_{f_a1}_{idx_a1}_{f_a2}_{idx_a2}_"
                    f"{f_b1}_{idx_b1}_{f_b2}_{idx_b2}.npz")
    cov_filename = os.path.join(args.output_path, cov_filename)
    print("Saving covariance matrices to ", cov_filename)
    np.savez(cov_filename, **cov_matrix)
