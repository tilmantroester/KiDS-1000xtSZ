import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import read_partial_map


def make_maps(nside, e1, e2, w, idx, rotate=False):
    n_pix = healpy.nside2npix(nside)

    if rotate:
        alpha = np.pi*np.random.rand(len(e1))
        e = np.sqrt(e1**2 + e2**2)
        e1 = np.cos(2.0*alpha)*e
        e2 = np.sin(2.0*alpha)*e

    e1_map = np.bincount(idx, weights=w*e1, minlength=n_pix)
    e2_map = np.bincount(idx, weights=w*e2, minlength=n_pix)
    w_map = np.bincount(idx, weights=w, minlength=n_pix)

    return e1_map, e2_map, w_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--Cl-coupled-filename")
    parser.add_argument("--Cl-decoupled-filename", required=True)

    parser.add_argument("--bin-operator", required=True)

    parser.add_argument("--shear-maps", nargs="+")
    parser.add_argument("--shear-masks", nargs="+")
    parser.add_argument("--shear-catalogs", nargs="+")

    parser.add_argument("--pymaster-workspace", required=True)

    parser.add_argument("--n-randoms", required=True)

    parser.add_argument("--n-iter")

    args = parser.parse_args()

    nside = 2048
    n_pix = healpy.nside2npix(nside)

    n_iter = 3
    if args.n_iter is not None:
        n_iter = int(args.n_iter)
        print(f"Using n_iter = {n_iter}")

    n_random = int(args.n_randoms)
    print(f"Producing {n_random} randoms")

    if args.shear_maps is not None and args.shear_catalogs is not None:
        raise ValueError("Either shear-maps or shear-catalogs "
                         "should be specified.")

    # Creating binning object
    if args.bin_operator.find("delta_ell_") == 0:
        delta_ell = int(args.bin_operator[len("delta_ell_"):])
        print("Using linear binning with bin width ", delta_ell)
        nmt_bins = nmt.NmtBin.from_nside_linear(
                        nside=nside,
                        nlb=delta_ell)
    else:
        print("Using binning operator from file ", args.bin_operator)
        binning_operator = np.loadtxt(args.bin_operator)
        ell = np.arange(binning_operator.size)

        nmt_bins = nmt.NmtBin(nside=nside,
                              bpws=binning_operator, ells=ell, weights=2*ell+1)

    ell_eff = nmt_bins.get_effective_ells()

    if args.Cl_coupled_filename:
        print("Saving coupled Cls to ", args.Cl_coupled_filename)
    if args.Cl_decoupled_filename:
        print("Saving decoupled Cls to ", args.Cl_decoupled_filename)
        output_path = os.path.split(args.Cl_decoupled_filename)[0]
        os.makedirs(output_path, exist_ok=True)

    # Loading workspace
    print("Loading workspace: ", args.pymaster_workspace)
    nmt_workspace = nmt.NmtWorkspace()
    nmt_workspace.read_from(args.pymaster_workspace)

    # Loading maps
    if args.shear_maps is not None:
        use_maps = True

        probes = ["A", "B"][:len(args.shear_maps)]
        w_map = {}
        e1_map = {}
        e2_map = {}
        for probe, shear_map_file, weight_map_file in zip(probes,
                                                          args.shear_maps,
                                                          args.shear_masks):
            print("Probe ", probe)
            print("  Loading shear map: ", shear_map_file)

            w_map[probe] = healpy.read_map(weight_map_file, verbose=False)
            w_map[probe][w_map[probe] == healpy.UNSEEN] = 0
            e1_map[probe], e2_map[probe] = read_partial_map(
                                                shear_map_file,
                                                fields=[2, 3], fill_value=0,
                                                scale=[1, 1])

    # Loading catalogs
    if args.shear_catalogs is not None:
        use_maps = False

        probes = ["A", "B"][:len(args.shear_catalogs)]
        shear_data = {}
        w_map = {}
        for probe, shear_catalog_file in zip(probes, args.shear_catalogs):
            print("Probe ", probe)
            print("  Loading shear catalog: ", shear_catalog_file)
            shear_data[probe] = np.load(shear_catalog_file)

    # Do the computations
    spectra = {"EE": 0, "EB": 1, "BE": 2, "BB": 3}

    Cls_coupled = {name: [] for name in spectra.keys()}
    Cls_decoupled = {name: [] for name in spectra.keys()}

    for i in range(n_random):
        print("Randoms ", i)
        field = {}
        for probe in probes:
            print("  Probe ", probe)
            if use_maps:
                alpha = np.pi*np.random.rand(n_pix)
                e_map = np.sqrt(e1_map[probe]**2 + e2_map[probe]**2)
                random_e1_map = np.cos(2.0*alpha)*e_map
                random_e2_map = np.sin(2.0*alpha)*e_map
            else:
                random_e1_map, random_e2_map, w_map[probe] = \
                                    make_maps(nside,
                                              -shear_data[probe]["e1"],
                                              shear_data[probe]["e2"],
                                              shear_data[probe]["w"],
                                              shear_data[probe]["pixel_idx"],
                                              rotate=True)

            print("    Creating field object")
            field[probe] = nmt.NmtField(w_map[probe],
                                        [random_e1_map, random_e2_map],
                                        n_iter=n_iter)
        if "B" not in field:
            field["B"] = field["A"]

        print("  Computing Cls")
        Cl_coupled = nmt.compute_coupled_cell(field["A"], field["B"])
        noise_bias = None
        Cl_decoupled = nmt_workspace.decouple_cell(cl_in=Cl_coupled,
                                                   cl_noise=noise_bias)

        for name, idx in spectra.items():
            Cls_coupled[name].append(Cl_coupled[idx])
            Cls_decoupled[name].append(Cl_decoupled[idx])

        if args.Cl_coupled_filename is not None:
            ell = np.arange(3*nside)
            np.savez(args.Cl_coupled_filename,
                     ell=ell,
                     **{name: np.array(Cls_coupled[name])
                        for name in spectra.keys()})

        if args.Cl_decoupled_filename is not None:
            np.savez(args.Cl_decoupled_filename,
                     ell=ell_eff,
                     **{name: np.array(Cls_decoupled[name])
                        for name in spectra.keys()})
