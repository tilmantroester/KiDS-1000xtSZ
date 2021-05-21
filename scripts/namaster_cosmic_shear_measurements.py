import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import read_partial_map, printflush


def make_maps(nside, e1, e2, w, idx, rotate=False, return_w2_sigma2=False):
    n_pix = healpy.nside2npix(nside)

    if rotate:
        alpha = np.pi*np.random.rand(len(e1))
        e = np.sqrt(e1**2 + e2**2)
        e1 = np.cos(2.0*alpha)*e
        e2 = np.sin(2.0*alpha)*e

    e1_map = np.bincount(idx, weights=w*e1, minlength=n_pix)
    e2_map = np.bincount(idx, weights=w*e2, minlength=n_pix)
    w_map = np.bincount(idx, weights=w, minlength=n_pix)

    good_pixel = w_map > 0
    e1_map[good_pixel] /= w_map[good_pixel]
    e2_map[good_pixel] /= w_map[good_pixel]

    if return_w2_sigma2:
        w2_sigma2_map = np.bincount(idx, weights=w**2*(e1**2 + e2**2)/2,
                                    minlength=n_pix)
        return e1_map, e2_map, w_map, w2_sigma2_map

    return e1_map, e2_map, w_map


def make_signal_map(nside, Cl_EE_signal, mask=None):
    Cl_0 = np.zeros(3*nside)

    if len(Cl_EE_signal) == 1:
        # 1 spin-2 field: nmaps = 2, ncls = 3
        # EE, EB, BB
        m = nmt.synfast_spherical(nside,
                                  cls=[Cl_EE_signal[("A", "A")],  # EE
                                       Cl_0,                      # EB
                                       Cl_0],                     # BB
                                  spin_arr=[2])
        if mask is not None:
            m[:, ~mask] = 0

        m = {"A": m}
    elif len(Cl_EE_signal) == 3:
        # 2 spin-2 fields: nmaps = 4, ncls = 10
        # E1E1, E1B1, E1E2, E1B2, B1B1, B1E2, B1B2, E2E2, E2B2, B2B2

        m = nmt.synfast_spherical(nside,
                                  cls=[Cl_EE_signal[("A", "A")],  # E1E1
                                       Cl_0,                      # E1B1
                                       Cl_EE_signal[("B", "A")],  # E1E2
                                       Cl_0,                      # E1B2
                                       Cl_0,                      # B1B1
                                       Cl_0,                      # B1E2
                                       Cl_0,                      # B1B2
                                       Cl_EE_signal[("B", "B")],  # E2E2
                                       Cl_0,                      # E2B2
                                       Cl_0],                     # B2B2
                                  spin_arr=[2, 2])
        if mask is not None:
            m[:, ~mask] = 0

        m = {"A": m[0:2], "B": m[2:4]}
    else:
        raise ValueError("Only one or two fields are supported right now.")

    return m


def make_signal_Cls(nside, nofz_files):
    import pyccl as ccl

    ccl_cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, sigma8=0.8,
                              n_s=0.97, h=0.7, m_nu=0.0)

    probes = ["A", "B"][:len(nofz_files)]
    WL_tracers = {}
    for filename, probe in zip(nofz_files, probes):
        z, nz = np.loadtxt(filename, unpack=True)
        z += 0.025
        WL_tracers[probe] = ccl.WeakLensingTracer(cosmo=ccl_cosmo,
                                                  dndz=(z, nz))

    ell = np.arange(3*nside)

    Cl_EE_signal = {}
    for i, probe_1 in enumerate(probes):
        for probe_2 in probes[:i+1]:
            Cl_EE_signal[(probe_1, probe_2)] = \
                ccl.angular_cl(cosmo=ccl_cosmo,
                               cltracer1=WL_tracers[probe_1],
                               cltracer2=WL_tracers[probe_2],
                               ell=ell)

    return Cl_EE_signal


def compute_Cl_cov(Cl_EE_signal, w_map_a, w_map_b, wsp):
    nside = healpy.get_nside(w_map_a)
    if len(Cl_EE_signal) == 1:
        Cl_EE = Cl_EE_signal[("A", "A")]
    else:
        # Use the cross-correlation
        Cl_EE = Cl_EE_signal[("B", "A")]

    Cl_0 = np.zeros(3*nside)

    Cls_coupled = wsp.couple_cell([Cl_EE, Cl_0, Cl_0, Cl_0])
    mean_w2 = (w_map_a*w_map_b).mean()
    return Cls_coupled/mean_w2


def compute_Cl_noise_bias(nside, mean_w2_sigma2):
    Cl_0 = np.zeros(3*nside)
    Cl_1 = np.ones(3*nside)
    pixel_area = healpy.nside2pixarea(nside)

    N_bias = pixel_area * mean_w2_sigma2
    noise_bias = [Cl_1*N_bias, Cl_0, Cl_0, Cl_1*N_bias]
    return noise_bias


def compute_Cl_noise_cov(nside, mean_w2_sigma2, w_map_a):
    Cl_noise_bias = np.array(compute_Cl_noise_bias(nside, mean_w2_sigma2))
    mean_w2 = (w_map_a**2).mean()
    Cl_noise_cov = Cl_noise_bias/mean_w2
    return Cl_noise_cov


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--Cl-coupled-filename")
    parser.add_argument("--Cl-decoupled-filename")
    parser.add_argument("--Cl-decoupled-no-noise-bias-filename")

    parser.add_argument("--Cl-data-filename")
    parser.add_argument("--Cl-cov-filename")

    parser.add_argument("--bin-operator", required=True)

    parser.add_argument("--shear-maps", nargs="+")
    parser.add_argument("--shear-masks", nargs="+")
    parser.add_argument("--shear-catalogs", nargs="+")

    parser.add_argument("--nofz-files", nargs="+")

    parser.add_argument("--pymaster-workspace", required=True)
    parser.add_argument("--bandpower-windows-filename")

    parser.add_argument("--n-randoms", required=True)

    parser.add_argument("--n-iter")

    parser.add_argument("--compute-coupling-matrix", action="store_true")
    parser.add_argument("--binary-mask", action="store_true")

    parser.add_argument("--no-flip-e1", action="store_true")
    parser.add_argument("--flip-e2", action="store_true")

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

    if args.nofz_files is not None:
        print("Adding signal maps to randoms.")
        print("  Creating signal Cls")
        Cl_EE_signal = make_signal_Cls(nside, args.nofz_files)
    else:
        Cl_EE_signal = None

    binary_mask = args.binary_mask
    if binary_mask:
        print("Using binary mask")

    flip_e1 = not args.no_flip_e1
    if flip_e1:
        print("Flipping sign of e1.")
    else:
        print("Not flipping sign of e1.")

    flip_e2 = args.flip_e2
    if flip_e2:
        print("Flipping sign of e2.")
    else:
        print("Not flipping sign of e2.")

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
        output_path = os.path.split(args.Cl_coupled_filename)[0]
        os.makedirs(output_path, exist_ok=True)
    if args.Cl_decoupled_filename:
        print("Saving decoupled Cls to ", args.Cl_decoupled_filename)
        output_path = os.path.split(args.Cl_decoupled_filename)[0]
        os.makedirs(output_path, exist_ok=True)
    if args.Cl_decoupled_no_noise_bias_filename:
        if len(args.shear_catalogs) > 1:
            raise ValueError("Computing cross-spectra, "
                             "no noise bias to subtact")
        print("Subtracting noise bias")
        print("Saving decoupled Cls with noise bias removed to ",
              args.Cl_decoupled_no_noise_bias_filename)
        output_path = os.path.split(
                                args.Cl_decoupled_no_noise_bias_filename)[0]
        os.makedirs(output_path, exist_ok=True)
        subtract_noise_bias = True
    else:
        subtract_noise_bias = False

    if args.Cl_data_filename is not None:
        compute_data_Cls = True
        print("Will compute data Cls and save them to ",
              args.Cl_data_filename)
        output_path = os.path.split(args.Cl_data_filename)[0]
        os.makedirs(output_path, exist_ok=True)
    else:
        compute_data_Cls = False

    if args.Cl_cov_filename is not None:
        print("Saving covariance Cls to ", args.Cl_cov_filename)
        output_path = os.path.split(args.Cl_cov_filename)[0]
        os.makedirs(output_path, exist_ok=True)

    if args.compute_coupling_matrix:
        nmt_workspace = None
        print("Will compute coupling matrix and save it to ",
              args.pymaster_workspace)
        output_path = os.path.split(args.pymaster_workspace)[0]
        os.makedirs(output_path, exist_ok=True)
    else:
        # Loading workspace
        print("Loading workspace: ", args.pymaster_workspace)
        nmt_workspace = nmt.NmtWorkspace()
        nmt_workspace.read_from(args.pymaster_workspace)

    if args.bandpower_windows_filename:
        print("Saving bandpower windows to ", args.bandpower_windows_filename)
        output_path = os.path.split(args.bandpower_windows_filename)[0]
        os.makedirs(output_path, exist_ok=True)

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
        good_pixel_map = {}
        mean_w2_sigma2 = {}
        field = {}

        e1_sign = -1 if flip_e1 else 1
        e2_sign = -1 if flip_e2 else 1

        for probe, shear_catalog_file in zip(probes, args.shear_catalogs):
            print("Probe ", probe)
            print("  Loading shear catalog: ", shear_catalog_file)
            shear_data[probe] = np.load(shear_catalog_file)
            e1_map, e2_map, w_map[probe], w2_s2_map = make_maps(
                                                nside,
                                                e1_sign*shear_data[probe]["e1"],
                                                e2_sign*shear_data[probe]["e2"],
                                                shear_data[probe]["w"],
                                                shear_data[probe]["pixel_idx"],
                                                return_w2_sigma2=True)
            good_pixel_map[probe] = w_map[probe] > 0

            if binary_mask:
                mean_w2_sigma2[probe] = np.sum(
                    w2_s2_map[good_pixel_map[probe]]
                    / w_map[good_pixel_map[probe]]**2
                    ) / n_pix
                w_map[probe] = good_pixel_map[probe]
            else:
                mean_w2_sigma2[probe] = w2_s2_map.sum()/n_pix

            if compute_data_Cls:
                print("  Creating field")
                field[probe] = nmt.NmtField(w_map[probe],
                                            [e1_map, e2_map],
                                            n_iter=n_iter)

    # Compute coupling matrix
    if nmt_workspace is None:
        print("Computing coupling matrix")
        if len(field) == 0:
            for probe in probes:
                print("  Creating field object")
                field[probe] = nmt.NmtField(w_map[probe],
                                            None, spin=2,
                                            n_iter=n_iter)
        if "B" not in field:
            field["B"] = field["A"]

        nmt_workspace = nmt.NmtWorkspace()
        nmt_workspace.compute_coupling_matrix(fl1=field["A"],
                                              fl2=field["B"],
                                              bins=nmt_bins,
                                              is_teb=False,
                                              n_iter=n_iter)
        nmt_workspace.write_to(args.pymaster_workspace)

    # Save baddpower windows
    if args.bandpower_windows_filename is not None:
        np.save(args.bandpower_windows_filename,
                nmt_workspace.get_bandpower_windows())

    # Compute data Cls
    if compute_data_Cls:
        if "B" not in field:
            field["B"] = field["A"]

        print("Computing data Cls")
        Cl_coupled = nmt.compute_coupled_cell(field["A"], field["B"])
        Cl_decoupled = nmt_workspace.decouple_cell(cl_in=Cl_coupled)

        if field["A"] == field["B"]:
            noise_bias = compute_Cl_noise_bias(nside, mean_w2_sigma2["A"])
            Cl_decoupled_no_noise_bias = nmt_workspace.decouple_cell(
                                                    cl_in=Cl_coupled,
                                                    cl_noise=noise_bias)
        else:
            Cl_decoupled_no_noise_bias = Cl_decoupled

        ell = np.arange(3*nside)
        np.savez(args.Cl_data_filename,
                 ell=ell,
                 Cl_coupled=Cl_coupled,
                 ell_eff=ell_eff,
                 Cl_decoupled=Cl_decoupled_no_noise_bias,
                 Cl_decoupled_raw=Cl_decoupled)

    # Compute covariance Cls
    if args.Cl_cov_filename is not None:
        printflush("Computing Cls for covariance")
        Cl_cov = compute_Cl_cov(
                            Cl_EE_signal,
                            w_map_a=w_map["A"],
                            w_map_b=w_map["B"] if "B" in w_map else w_map["A"],
                            wsp=nmt_workspace)
        if len(probes) == 1:
            # Auto correlation
            Cl_noise_cov = compute_Cl_noise_cov(nside,
                                                mean_w2_sigma2["A"],
                                                w_map["A"])
        else:
            # Cross correlation
            Cl_noise_cov = np.zeros_like(Cl_cov)

        np.savez(args.Cl_cov_filename,
                 ell=np.arange(Cl_cov.shape[1]),
                 Cl_cov=Cl_cov, Cl_noise_cov=Cl_noise_cov)

    # Do the randoms computations
    spectra = {"EE": 0, "EB": 1, "BE": 2, "BB": 3}

    Cls_coupled = {name: [] for name in spectra.keys()}
    Cls_decoupled = {name: [] for name in spectra.keys()}
    Cls_decoupled_no_noise_bias = {name: [] for name in spectra.keys()}

    for i in range(n_random):
        print("Randoms ", i)
        field = {}
        if Cl_EE_signal is not None:
            print("  Creating signal maps")
            signal_maps = make_signal_map(nside, Cl_EE_signal,
                                          mask=w_map[probe] > 0)

        for probe in probes:
            print("  Probe ", probe)
            if use_maps:
                alpha = np.pi*np.random.rand(n_pix)
                e_map = np.sqrt(e1_map[probe]**2 + e2_map[probe]**2)
                random_e1_map = np.cos(2.0*alpha)*e_map
                random_e2_map = np.sin(2.0*alpha)*e_map
            else:
                random_e1_map, random_e2_map, _ = \
                                    make_maps(nside,
                                              e1_sign*shear_data[probe]["e1"],
                                              e2_sign*shear_data[probe]["e2"],
                                              shear_data[probe]["w"],
                                              shear_data[probe]["pixel_idx"],
                                              rotate=True)

            if Cl_EE_signal is not None:
                random_e1_map += signal_maps[probe][0]
                random_e2_map += signal_maps[probe][1]

            print("    Creating field object")
            field[probe] = nmt.NmtField(w_map[probe],
                                        [random_e1_map, random_e2_map],
                                        n_iter=n_iter)
        if "B" not in field:
            field["B"] = field["A"]

        print("  Computing Cls")
        Cl_coupled = nmt.compute_coupled_cell(field["A"], field["B"])
        Cl_decoupled = nmt_workspace.decouple_cell(cl_in=Cl_coupled)

        if len(probes) == 1 and subtract_noise_bias:
            noise_bias = compute_Cl_noise_bias(nside, mean_w2_sigma2["A"])
            Cl_decoupled_no_noise_bias = nmt_workspace.decouple_cell(
                                                    cl_in=Cl_coupled,
                                                    cl_noise=noise_bias)

        for name, idx in spectra.items():
            Cls_coupled[name].append(Cl_coupled[idx])
            Cls_decoupled[name].append(Cl_decoupled[idx])
            if subtract_noise_bias:
                Cls_decoupled_no_noise_bias[name].append(
                                            Cl_decoupled_no_noise_bias[idx])

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

        if args.Cl_decoupled_no_noise_bias_filename is not None:
            np.savez(args.Cl_decoupled_no_noise_bias_filename,
                     ell=ell_eff,
                     **{name: np.array(Cls_decoupled_no_noise_bias[name])
                        for name in spectra.keys()})
