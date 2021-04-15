import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import read_partial_map, file_header


def compute_master(f_a, f_b, wsp):
    # Compute the power spectrum (a la anafast) of the masked fields
    # Note that we only use n_iter=0 here to speed up the computation,
    # but the default value of 3 is recommended in general.
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    # Decouple power spectrum into bandpowers inverting the coupling matrix
    cl_decoupled = wsp.decouple_cell(cl_coupled)

    return cl_decoupled


def bin_theory(cl, wsp):
    return wsp.decouple_cell(wsp.couple_cell(cl))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--bin-operator", required=True)

    parser.add_argument("--shear-maps", nargs="+", required=True)
    parser.add_argument("--shear-masks", nargs="+", required=True)

    parser.add_argument("--foreground-map", required=True)
    parser.add_argument("--foreground-mask", required=True)

    parser.add_argument("--foreground-mask-already-applied",
                        action="store_true")

    parser.add_argument("--pymaster-workspace-path")

    args = parser.parse_args()

    n_iter = 3

    os.makedirs(args.output_path, exist_ok=True)

    if len(args.shear_maps) != len(args.shear_masks):
        raise ValueError("Number of shear masks does not match number of "
                         "shear masks.")

    shear_fields = []
    print("Loading shear maps")
    for shear_map_file, mask_file in zip(args.shear_maps, args.shear_masks):
        print(shear_map_file)
        shear_mask = healpy.read_map(mask_file, verbose=False)
        shear_mask[shear_mask == healpy.UNSEEN] = 0

        shear_data = read_partial_map(shear_map_file,
                                      fields=[2, 3], fill_value=0,
                                      scale=[1, 1])

        print("  Creating field object")
        field_background = nmt.NmtField(shear_mask,
                                        shear_data,
                                        n_iter=n_iter)

        shear_fields.append(field_background)

    print("Loading foreground map")
    print(args.foreground_map)
    if args.foreground_mask_already_applied:
        print("  Mask already applied to map.")
    foreground_map = healpy.read_map(args.foreground_map, verbose=False)
    foreground_map[foreground_map == healpy.UNSEEN] = 0

    foreground_mask = healpy.read_map(args.foreground_mask, verbose=False)
    foreground_mask[foreground_mask == healpy.UNSEEN] = 0

    print("  Creating field object")
    foreground_field = nmt.NmtField(
                        foreground_mask,
                        [foreground_map],
                        masked_on_input=args.foreground_mask_already_applied,
                        n_iter=n_iter)

    binning_operator = np.loadtxt(args.bin_operator)
    ell = np.arange(binning_operator.size)

    nmt_bins = nmt.NmtBin(nside=healpy.get_nside(foreground_map),
                          bpws=binning_operator, ells=ell, weights=2*ell+1)

    nmt_workspaces = []
    if args.pymaster_workspace_path is None:
        print("Creating workspaces and computing coupling matrices")
        for i, shear_field in enumerate(shear_fields):
            print(f"  Field {i}")
            nmt_workspace = nmt.NmtWorkspace()
            nmt_workspace.compute_coupling_matrix(fl1=foreground_field,
                                                  fl2=shear_field,
                                                  bins=nmt_bins,
                                                  is_teb=False,
                                                  n_iter=n_iter)
            nmt_workspaces.append(nmt_workspace)

            nmt_workspace.write_to(os.path.join(
                            args.output_path,
                            f"pymaster_workspace_foreground_shear_{i}.fits"))
    else:
        print("Reading existing workspaces")
        for i in range(len(shear_fields)):
            print(f"  Field {i}")
            nmt_workspace = nmt.NmtWorkspace()
            nmt_workspace.read_from(os.path.join(
                            args.pymaster_workspace_path,
                            f"pymaster_workspace_foreground_shear_{i}.fits"))
            nmt_workspaces.append(nmt_workspace)

    print("Computing Cls")
    Cls_coupled = []
    Cls_decoupled = []
    for i, (shear_field, nmt_workspace) in \
            enumerate(zip(shear_fields, nmt_workspaces)):
        print(f"  Field {i}")
        Cl_coupled = nmt.compute_coupled_cell(foreground_field, shear_field)
        Cl_decoupled = nmt_workspace.decouple_cell(Cl_coupled)

        Cls_coupled.append(Cl_coupled)
        Cls_decoupled.append(Cl_decoupled)

    ell_nmt = nmt_bins.get_effective_ells()
    header = "ell, " + ", ".join([f"Cl_TE_zbin_{i}, Cl_TB_zbin_{i}"
                                  for i in range(len(shear_fields))])
    header = file_header(header_info=header)

    np.savetxt(os.path.join(args.output_path, "Cl_decoupled.txt"),
               np.vstack((ell_nmt, *Cls_decoupled)).T,
               header=header
               )

    ell_coupled = np.arange(Cls_coupled[0].shape[1])
    np.savetxt(os.path.join(args.output_path, "Cl_coupled.txt"),
               np.vstack((ell_coupled, *Cls_coupled)).T,
               header=header
               )

    print("Computing coupling matricies for Gaussian covariance")

    for i, shear_field_a in enumerate(shear_fields):
        for j, shear_field_b in enumerate(shear_fields[:i+1]):
            print(f"  Field {i}-{j}")
            nmt_cov_workspace = nmt.NmtCovarianceWorkspace()

            nmt_cov_workspace.compute_coupling_coefficients(
                                          fla1=foreground_field,
                                          fla2=shear_field_a,
                                          flb1=foreground_field,
                                          flb2=shear_field_b)

            nmt_cov_workspace.write_to(
                os.path.join(
                    args.output_path,
                    f"pymaster_cov_workspace_foreground_shear_{i}"
                    f"_foreground_shear_{j}.fits"))
