import os
import copy

import numpy as np

import scipy.interpolate

import pyccl as ccl

import pyhmcode

import misc_utils

PI = np.pi


def load_cosmosis_params(output_path, section="cosmological_parameters"):
    with open(os.path.join(output_path, section, "values.txt"), "r") as f:
        params = {line.split("=")[0].strip(): float(line.split("=")[1].strip())
                  for line in f.readlines()}
    return params


class HaloProfileInterpolated(ccl.halos.HaloProfile):
    def __init__(self, interpolator, is_logk=True, is_logM=True,
                 is_logp=True, norm=None):
        self.interpolator = interpolator
        self.is_logk = is_logk
        self.is_logM = is_logM
        self.is_logp = is_logp

        self.norm = norm or 1

    def _fourier(self, cosmo, k, M, a, mass_def):
        k_h = k/cosmo["h"]
        M_h = M*cosmo["h"]

        k_h = self._check_shape(k_h)
        M_h = self._check_shape(M_h)
        a = self._check_shape(a)

        k_h = np.log(k_h) if self.is_logk else k_h
        M_h = np.log(M_h) if self.is_logM else M_h

        positions = np.vstack([m.ravel() for m in np.meshgrid(a, k_h, M_h)]).T

        profile = self.interpolator(positions).reshape(len(k_h), len(M_h)).T

        if self.is_logp:
            profile = np.exp(profile)

        profile *= self.norm

        # Turn Mpc^3 h^-3 into Mpc^3
        profile *= cosmo["h"]**-3

        return profile.squeeze()

    def _check_shape(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if arr.ndim < 1:
            arr = arr[None]
        return arr


class HMxProfileGenerator:
    def __init__(self, hmcode_cosmo, a_arr, k_arr, verbose=False):
        self.hmcode_cosmo = hmcode_cosmo
        self.hmod = pyhmcode.Halomodel(
                        pyhmcode.HMx2020_matter_pressure_w_temp_scaling,
                        verbose=verbose)

        self.a_arr = a_arr.copy()
        self.k_arr = k_arr.copy()

        self.verbose = verbose

        self.fields = [pyhmcode.field_matter,
                       pyhmcode.field_cdm,
                       pyhmcode.field_electron_pressure]

        self._has_interpolator = False
        self._compute_tables()

    def _compute_tables(self):
        if self.verbose:
            print("Computing profile look-up table")

        profile_lut = np.empty((len(self.fields),
                                len(self.a_arr),
                                len(self.k_arr),
                                self.hmod.n), dtype=np.float64)

        for a_idx, a in enumerate(self.a_arr):
            if self.verbose:
                print("Computing a = ", a)
            pyhmcode.hmx.init_halomod(a, self.hmod,
                                      self.hmcode_cosmo, self.verbose)
            for k_idx, k in enumerate(self.k_arr):
                wk = np.empty((self.hmod.n, len(self.fields)), order="f")
                pyhmcode.hmx.init_windows(k, self.fields, len(self.fields),
                                          wk, self.hmod.n,
                                          self.hmod, self.hmcode_cosmo)
                profile_lut[:, a_idx, k_idx] = wk.T

        # Turn electron pressure into physical units
        pressure_idx = self.fields.index(pyhmcode.field_electron_pressure)
        profile_lut[pressure_idx] *= self.a_arr[:, None, None]**-3

        profile_lut[profile_lut <= 1e-30] = 1e-30

        profile_interpolator = [scipy.interpolate.RegularGridInterpolator(
                                            points=(self.a_arr,
                                                    np.log(self.k_arr),
                                                    np.log(self.hmod.m)),
                                            values=np.log(profile_lut[f_idx]),
                                            method="linear",
                                            bounds_error=True, fill_value=None)
                                for f_idx, f in enumerate(self.fields)]

        self.profiles = {f: HaloProfileInterpolated(
                                    profile_interpolator[f_idx],
                                    is_logk=True, is_logM=True, is_logp=True,
                                    norm=self.field_normalisation(f))
                         for f_idx, f in enumerate(self.fields)}

        self._has_interpolator = True

    def field_normalisation(self, field):
        if field == pyhmcode.field_cdm:
            return self.hmcode_cosmo.om_m/self.hmcode_cosmo.om_c
        else:
            return 1.0

    @property
    def matter_profile(self):
        if pyhmcode.field_matter in self.fields:
            return self.profiles[pyhmcode.field_matter]
        else:
            raise ValueError("Matter profiles not computed")

    @property
    def cdm_profile(self):
        if pyhmcode.field_cdm in self.fields:
            return self.profiles[pyhmcode.field_cdm]
        else:
            raise ValueError("CDM profiles not computed")

    @property
    def pressure_profile(self):
        if pyhmcode.field_electron_pressure in self.fields:
            return self.profiles[pyhmcode.field_electron_pressure]
        else:
            raise ValueError("Pressure profiles not computed")


class CovarianceCalculator:
    def __init__(self, ccl_cosmo, lensing_nz, pofk_lin=None):
        self.ccl_cosmo = ccl_cosmo

        self.WL_tracers = []
        z_max = 0.0
        for (z, nz) in lensing_nz:
            self.WL_tracers.append(ccl.WeakLensingTracer(
                                            self.ccl_cosmo,
                                            dndz=(z, nz)))
            z_max = max(z_max, nz.max())

        self.n_z = len(self.WL_tracers)

        self.tSZ_tracers = [ccl.tSZTracer(self.ccl_cosmo, z_max=z_max)]

        if pofk_lin is not None:
            self.pofk_lin, self.pofk_lin_k_h, self.pofk_lin_z = pofk_lin

        self._has_halo_profiles = False

    def compute_halo_profiles(self, a_arr, k_arr, verbose=True):
        hmcode_cosmo = pyhmcode.Cosmology()

        hmcode_cosmo.om_m = self.ccl_cosmo["Omega_m"]
        hmcode_cosmo.om_b = self.ccl_cosmo["Omega_b"]
        hmcode_cosmo.om_v = self.ccl_cosmo["Omega_l"]
        hmcode_cosmo.h = self.ccl_cosmo["h"]
        hmcode_cosmo.ns = self.ccl_cosmo["n_s"]
        sigma8 = self.ccl_cosmo["sigma8"]
        if not np.isfinite(sigma8):
            sigma8 = ccl.sigma8(self.ccl_cosmo)
        hmcode_cosmo.sig8 = sigma8
        hmcode_cosmo.m_nu = self.ccl_cosmo["m_nu"].sum()

        hmcode_cosmo.set_linear_power_spectrum(self.pofk_lin_k_h,
                                               self.pofk_lin_z,
                                               self.pofk_lin)

        self.halo_profiles = HMxProfileGenerator(hmcode_cosmo,
                                                 a_arr=a_arr, k_arr=k_arr/self.ccl_cosmo["h"],
                                                 verbose=verbose)

        self._has_halo_profiles = True

    def compute_trispectra(self, probes, a_arr, k_arr,
                           pofk_background_linear=None,
                           cNG=True, SSC=True,
                           halo_model="hmx", verbose=True):

        profiles = {}

        if halo_model == "hmx":
            if not self._has_halo_profiles:
                self.compute_halo_profiles(a_arr, k_arr,
                                           verbose=verbose)

            mass_def = ccl.halos.MassDef("vir", 'matter')
            hmf = ccl.halos.MassFuncSheth99(self.ccl_cosmo,
                                            mass_def=mass_def,
                                            mass_def_strict=False,
                                            use_delta_c_fit=True)
            hbf = ccl.halos.HaloBiasSheth99(self.ccl_cosmo,
                                            mass_def=mass_def,
                                            mass_def_strict=False,
                                            use_delta_c_fit=True)
            hmc = ccl.halos.HMCalculator(self.ccl_cosmo, hmf, hbf, mass_def)
            profiles["matter"] = self.halo_profiles.matter_profile
            profiles["pressure"] = self.halo_profiles.pressure_profile
            normalize_profile = False
        elif halo_model == "KiDS-1000":
            p = []
            for p1, p2 in probes:
                p += list(p1) + list(p2)
            if "pressure" in p:
                raise ValueError("halo_model KiDS-1000 only works "
                                 "for matter-matter.")

            mass_def = ccl.halos.MassDef200m()
            hmf = ccl.halos.MassFuncTinker10(self.ccl_cosmo,
                                             mass_def=mass_def)
            hbf = ccl.halos.HaloBiasTinker10(self.ccl_cosmo,
                                             mass_def=mass_def)
            hmc = ccl.halos.HMCalculator(self.ccl_cosmo, hmf, hbf, mass_def)
            profiles["matter"] = ccl.halos.HaloProfileNFW(
                            ccl.halos.ConcentrationDuffy08(mass_def),
                            fourier_analytic=True)
            normalize_profile = True
        else:
            raise ValueError(f"halo_model == {halo_model} not suppported.")

        if verbose:
            print("Computing halo model trispectrum")

        if cNG:
            self.tk3D_cNG = {}
        if SSC:
            self.tk3D_SSC = {}
        for (a1, a2), (b1, b2) in probes:
            if cNG:
                self.tk3D_cNG[(a1, a2), (b1, b2)] = ccl.halos.halomod_Tk3D_1h(
                                    cosmo=self.ccl_cosmo, hmc=hmc,
                                    prof1=profiles[a1],
                                    prof2=profiles[a2],
                                    prof12_2pt=None,
                                    prof3=profiles[b1],
                                    prof4=profiles[b2],
                                    prof34_2pt=None,
                                    normprof1=normalize_profile,
                                    normprof2=normalize_profile,
                                    normprof3=normalize_profile,
                                    normprof4=normalize_profile,
                                    lk_arr=np.log(k_arr), a_arr=a_arr,
                                    use_log=True)
            if SSC:
                self.tk3D_SSC[(a1, a2), (b1, b2)] = ccl.halos.halomod_Tk3D_SSC(
                                    cosmo=self.ccl_cosmo, hmc=hmc,
                                    prof1=profiles[a1],
                                    prof2=profiles[a2],
                                    prof12_2pt=None,
                                    prof3=profiles[b1],
                                    prof4=profiles[b2],
                                    prof34_2pt=None,
                                    p_of_k_a=pofk_background_linear,
                                    normprof1=normalize_profile,
                                    normprof2=normalize_profile,
                                    normprof3=normalize_profile,
                                    normprof4=normalize_profile,
                                    lk_arr=np.log(k_arr), a_arr=a_arr,
                                    use_log=True)

    def _compute_NG_covariance_block(self, mode,
                                     tr_a1, tr_a2, tr_b1, tr_b2, probe_3D,
                                     binning_operator_a, binning_operator_b,
                                     fsky=None, sigma2_B=None,
                                     beam_operator_a=None,
                                     beam_operator_b=None,
                                     n_ell_intp=100):

        ell_a = np.arange(binning_operator_a.shape[1])
        ell_a_coarse = np.linspace(2, ell_a[-1], n_ell_intp)

        ell_b = np.arange(binning_operator_b.shape[1])
        ell_b_coarse = np.linspace(2, ell_b[-1], n_ell_intp)

        if beam_operator_a is None:
            beam_operator_a = np.eye(len(ell_a))
        if beam_operator_b is None:
            beam_operator_b = np.eye(len(ell_b))

        if mode == "cNG":
            cov_coarse = ccl.angular_cl_cov_cNG(
                                        cosmo=self.ccl_cosmo,
                                        cltracer1=tr_a1, cltracer2=tr_a2,
                                        cltracer3=tr_b1, cltracer4=tr_b2,
                                        ell=ell_a_coarse, ell2=ell_b_coarse,
                                        tkka=self.tk3D_cNG[probe_3D],
                                        fsky=fsky)
        elif mode == "SSC":
            cov_coarse = ccl.angular_cl_cov_SSC(
                                        cosmo=self.ccl_cosmo,
                                        cltracer1=tr_a1, cltracer2=tr_a2,
                                        cltracer3=tr_b1, cltracer4=tr_b2,
                                        ell=ell_a_coarse, ell2=ell_b_coarse,
                                        tkka=self.tk3D_SSC[probe_3D],
                                        sigma2_B=sigma2_B)

        cov = np.exp(scipy.interpolate.RectBivariateSpline(
                                            ell_b_coarse, ell_a_coarse,
                                            np.log(cov_coarse))(ell_b, ell_a))

        cov = np.einsum("ij,jk,kl->il",
                        beam_operator_b, cov, beam_operator_a,
                        optimize='optimal')
        cov = np.einsum("bj,jk,ck->bc",
                        binning_operator_b, cov, binning_operator_a,
                        optimize='optimal')

        return cov

    def compute_NG_covariance(self, mode,
                              cov_blocks,
                              binning_operator,
                              beam_operator=None,
                              fsky=None, mask_wl=None,
                              halo_model="hmx",
                              k_arr=None, a_arr=None,
                              n_ell_intp=100, verbose=True):
        if k_arr is None:
            k_arr = np.logspace(-3, 1, 100)
        if a_arr is None:
            a_arr = np.linspace(1/(1+6), 1, 50)

        if mode == "SSC":
            if mask_wl is not None:
                if len(mask_wl) != 2:
                    mask_wl = (np.arange(len(mask_wl)), mask_wl)
                a = a_arr.copy()
                if a_arr[-1] == 1.0:
                    a[-1] = 1 - 1e-4
                sigma2_B = (a, [ccl.covariances.sigma2_B_from_mask_cl(
                                                self.ccl_cosmo,
                                                a=a_,
                                                mask_cl=mask_wl)
                                for a_ in a])
                if a_arr[-1] == 1.0:
                    sigma2_B = (np.append(a, 1.0),
                                np.append(sigma2_B[1][:-1], sigma2_B[1][-1]))
            else:
                sigma2_B = (a_arr, ccl.covariances.sigma2_B_disc(
                                                    self.ccl_cosmo,
                                                    a=a_arr, fsky=fsky))
        else:
            sigma2_B = None

        probes_3D = {}
        for cov_block in cov_blocks:
            if cov_block == "EEEE":
                probes_3D["EEEE"] = (("matter", "matter"),
                                     ("matter", "matter"))
            elif cov_block == "TETE":
                probes_3D["TETE"] = (("matter", "pressure"),
                                     ("matter", "pressure"))
            elif cov_block == "EETE":
                probes_3D["EETE"] = (("matter", "matter"),
                                     ("matter", "pressure"))

        self.compute_trispectra(
                            probes=probes_3D.values(),
                            a_arr=a_arr,
                            k_arr=k_arr,
                            pofk_background_linear=None,
                            cNG=(mode == "cNG"),
                            SSC=(mode == "SSC"),
                            halo_model=halo_model,
                            verbose=verbose)

        field_idx_EE = [(i, j) for i in range(self.n_z)
                        for j in range(i+1)]
        n_ell_bin_EE = binning_operator["EE"].shape[0]
        n_EE = n_ell_bin_EE*len(field_idx_EE)

        if any(["TE" in c for c in cov_blocks]):
            field_idx_TE = [(i, 0) for i in range(self.n_z)]
            n_ell_bin_TE = binning_operator["TE"].shape[0]
            n_TE = n_ell_bin_TE*len(field_idx_TE)

        cov = {}
        for cov_block in cov_blocks:
            if cov_block == "EEEE":
                cov["EEEE"] = np.zeros((n_EE, n_EE))
                for i, (idx_a1, idx_a2) in enumerate(field_idx_EE):
                    for j, (idx_b1, idx_b2) in enumerate(field_idx_EE[:i+1]):
                        print(cov_block, i, j)
                        c = self._compute_NG_covariance_block(
                                    mode,
                                    self.WL_tracers[idx_a1],
                                    self.WL_tracers[idx_a2],
                                    self.WL_tracers[idx_b1],
                                    self.WL_tracers[idx_b2],
                                    probe_3D=probes_3D[cov_block],
                                    binning_operator_a=binning_operator["EE"],
                                    binning_operator_b=binning_operator["EE"],
                                    fsky=fsky, sigma2_B=sigma2_B,
                                    beam_operator_a=None,
                                    beam_operator_b=None,
                                    n_ell_intp=n_ell_intp)
                        cov["EEEE"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                    j*n_ell_bin_EE:(j+1)*n_ell_bin_EE] = c.T
            elif cov_block == "TETE":
                cov["TETE"] = np.zeros((n_TE, n_TE))
                for i, (idx_a1, idx_a2) in enumerate(field_idx_TE):
                    for j, (idx_b1, idx_b2) in enumerate(field_idx_TE[:i+1]):
                        print(cov_block, i, j)
                        c = self._compute_NG_covariance_block(
                                    mode,
                                    self.WL_tracers[idx_a1],
                                    self.tSZ_tracers[idx_a2],
                                    self.WL_tracers[idx_b1],
                                    self.tSZ_tracers[idx_b2],
                                    probe_3D=probes_3D[cov_block],
                                    binning_operator_a=binning_operator["TE"],
                                    binning_operator_b=binning_operator["TE"],
                                    fsky=fsky, sigma2_B=sigma2_B,
                                    beam_operator_a=beam_operator,
                                    beam_operator_b=beam_operator,
                                    n_ell_intp=n_ell_intp)
                        cov["TETE"][i*n_ell_bin_TE:(i+1)*n_ell_bin_TE,
                                    j*n_ell_bin_TE:(j+1)*n_ell_bin_TE] = c.T
            elif cov_block == "EETE":
                cov["EETE"] = np.zeros((n_EE, n_TE))
                for i, (idx_a1, idx_a2) in enumerate(field_idx_EE):
                    for j, (idx_b1, idx_b2) in enumerate(field_idx_TE):
                        print(cov_block, i, j)
                        c = self._compute_NG_covariance_block(
                                    mode,
                                    self.WL_tracers[idx_a1],
                                    self.WL_tracers[idx_a2],
                                    self.WL_tracers[idx_b1],
                                    self.tSZ_tracers[idx_b2],
                                    probe_3D=probes_3D[cov_block],
                                    binning_operator_a=binning_operator["EE"],
                                    binning_operator_b=binning_operator["TE"],
                                    fsky=fsky, sigma2_B=sigma2_B,
                                    beam_operator_a=None,
                                    beam_operator_b=beam_operator,
                                    n_ell_intp=n_ell_intp)
                        cov["EETE"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                    j*n_ell_bin_TE:(j+1)*n_ell_bin_TE] = c.T

        if "joint" in cov_blocks:
            cov["joint"] = np.zeros((n_EE+n_TE, n_EE+n_TE))

            cov["joint"][:n_EE, :n_EE] = cov["EEEE"]
            cov["joint"][n_EE:n_EE+n_TE, :n_EE] = cov["EETE"].T
            cov["joint"][n_EE:n_EE+n_TE, n_EE:n_EE+n_TE] = cov["TETE"]

        for cov_block in cov_blocks:
            c = cov[cov_block]
            cov[cov_block][np.triu_indices_from(c, k=1)] = \
                c.T[np.triu_indices_from(c, k=1)]

        return cov

    def compute_m_covariance(self, theory_cl_path, m):
        field_idx_EE = [(i, j) for i in range(self.n_z)
                        for j in range(i+1)]
        Cl = []
        for i, j in field_idx_EE:
            d = np.loadtxt(os.path.join(theory_cl_path, f"bin_{i+1}_{j+1}.txt"))
            Cl.append(d)

        Cl = np.concatenate(Cl)
        Cl_matrix = np.outer(Cl, Cl)
        n_ell = d.size
        m_matrix = np.zeros((Cl.size, Cl.size))
        for i, (idx_a1, idx_a2) in enumerate(field_idx_EE):
            for j, (idx_b1, idx_b2) in enumerate(field_idx_EE):
                m_matrix[i*n_ell: (i+1)*n_ell, j*n_ell: (j+1)*n_ell] = (
                    m[idx_a1]*m[idx_b1] + m[idx_a1]*m[idx_b2]
                    + m[idx_a2]*m[idx_b1] + m[idx_a2]*m[idx_b2])

        return {"EEEE": m_matrix*Cl_matrix}


if __name__ == "__main__":
    prediction_path = "../runs/cov_theory_predictions_run2_beam10/output/data_block/"  # noqa: E501
    params = load_cosmosis_params(prediction_path)

    pofk_lin = np.loadtxt(os.path.join(prediction_path, "matter_power_lin/p_k.txt"))
    pofk_lin_z = np.loadtxt(os.path.join(prediction_path, "matter_power_lin/z.txt"))
    pofk_lin_k = np.loadtxt(os.path.join(prediction_path, "matter_power_lin/k_h.txt"))

    pofk_lin = (pofk_lin, pofk_lin_k, pofk_lin_z)

    ccl_cosmo = ccl.Cosmology(Omega_c=params["omega_c"],
                              Omega_b=params["omega_b"],
                              Omega_k=params["omega_k"],
                              n_s=params["n_s"],
                              sigma8=params["sigma_8"],
                              h=params["h0"],
                              m_nu=params["mnu"])

    b = np.loadtxt("../data/xcorr/bin_operator_log_n_bin_12_ell_51-2952.txt")
    binning_operator = {"EE": b, "TE": b}

    mask_wl = {"EE": np.loadtxt("../data/xcorr/cov/W_l/shear_KiDS1000_binary_auto.txt", unpack=True)}  # noqa: E501

    ell = np.arange(b.shape[1])
    beam_operator = misc_utils.create_beam_operator(ell,
                                                    fwhm=10.0,
                                                    # fwhm_map=10.0,
                                                    # fwhm_target=1.6
                                                    )

    z = np.loadtxt(os.path.join(prediction_path, "nz_kids1000/z.txt"))
    lensing_nz = [(z, np.loadtxt(os.path.join(prediction_path, f"nz_kids1000/bin_{i+1}.txt")))  # noqa: E501
                  for i in range(5)]

    cov_calculator = CovarianceCalculator(ccl_cosmo, lensing_nz, pofk_lin)

    cov = {}
    # cov["KiDS-1000xtSZ SSC disc"] = cov_calculator.compute_NG_covariance(
    #                         mode="SSC", cov_blocks=["TETE", "EETE", "EEEE", "joint"],
    #                         binning_operator=binning_operator,
    #                         beam_operator=beam_operator,
    #                         fsky=0.0216, 
    #                         halo_model="hmx",
    #                         n_ell_intp=100)
    
    # cov["KiDS-1000xtSZ cNG"] = cov_calculator.compute_NG_covariance(
    #                         mode="cNG", cov_blocks=["TETE", "EETE", "EEEE", "joint"],
    #                         binning_operator=binning_operator,
    #                         beam_operator=beam_operator,
    #                         fsky=0.0216, 
    #                         halo_model="hmx",
    #                         n_ell_intp=100)

    # header = misc_utils.file_header(
    #                     f"Covariance HMx EEEE SSC, disc geometry fsky=0.0216")
    # np.savetxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_SSC_disc_EE.txt",
    #            cov["KiDS-1000xtSZ SSC disc"]["EEEE"], header=header)
    
    # header = misc_utils.file_header(
    #                     f"Covariance HMx TETE SSC, disc geometry fsky=0.0216")
    # np.savetxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_SSC_disc_TE.txt",
    #            cov["KiDS-1000xtSZ SSC disc"]["TETE"], header=header)
    
    # header = misc_utils.file_header(
    #                     f"Covariance HMx joint EE+TE SSC, disc geometry fsky=0.0216")
    # np.savetxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_SSC_disc_EE+TE.txt",
    #            cov["KiDS-1000xtSZ SSC disc"]["joint"], header=header)

    # header = misc_utils.file_header(
    #                     f"Covariance HMx EEEE cNG, 1h fsky=0.0216")
    # np.savetxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_cNG_1h_EE.txt",
    #            cov["KiDS-1000xtSZ cNG"]["EEEE"], header=header)
    
    # header = misc_utils.file_header(
    #                     f"Covariance HMx TETE cNG, 1h fsky=0.0216")
    # np.savetxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_cNG_1h_TE.txt",
    #            cov["KiDS-1000xtSZ cNG"]["TETE"], header=header)
    
    # header = misc_utils.file_header(
    #                     f"Covariance HMx joint cNG, 1h fsky=0.0216")
    # np.savetxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_cNG_1h_EE+TE.txt",
    #            cov["KiDS-1000xtSZ cNG"]["joint"], header=header)


    cov["KiDS-1000xtSZ SSC disc"] = {"TETE": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_SSC_disc_TE.txt"),
                                     "joint": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_SSC_disc_EE+TE.txt")}
    cov["KiDS-1000xtSZ cNG"] = {"TETE": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_cNG_1h_TE.txt"),
                                     "joint": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_cNG_1h_EE+TE.txt")}
    cov["KiDS-1000xtSZ gaussian"] = {"TETE": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_gaussian_TE.txt"),
                                     "joint": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_gaussian_EE-TE.txt")}

    header = misc_utils.file_header(
                        f"Covariance HMx TETE Gaussian, SSC disc, cNG, 1h fsky=0.0216")
    np.savetxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_gaussian_SSC_disc_cNG_1h_TE.txt",
               (cov["KiDS-1000xtSZ gaussian"]["TETE"]
                + cov["KiDS-1000xtSZ SSC disc"]["TETE"]
                + cov["KiDS-1000xtSZ cNG"]["TETE"]), header=header)

    header = misc_utils.file_header(
                        f"Covariance HMx joint Gaussian, SSC disc, cNG, 1h fsky=0.0216")
    np.savetxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_hmx_gaussian_SSC_disc_cNG_1h_EE+TE.txt",
               (cov["KiDS-1000xtSZ gaussian"]["joint"]
                + cov["KiDS-1000xtSZ SSC disc"]["joint"]
                + cov["KiDS-1000xtSZ cNG"]["joint"]), header=header)
    # cov["KiDS-1000 SSC disc"] = cov_calculator.compute_NG_covariance(
    #                         mode="SSC", cov_blocks=["EEEE"],
    #                         binning_operator=binning_operator,
    #                         beam_operator=None,
    #                         fsky=0.0216, 
    #                         halo_model="KiDS-1000",
    #                         n_ell_intp=100)
    # cov["KiDS-1000 SSC mask"] = cov_calculator.compute_NG_covariance(
    #                         mode="SSC", cov_blocks=["EEEE"],
    #                         binning_operator=binning_operator,
    #                         beam_operator=None,
    #                         mask_wl=mask_wl["EE"],
    #                         halo_model="KiDS-1000",
    #                         n_ell_intp=100)
    
    # header = misc_utils.file_header(
    #                     f"Covariance EEEE SSC, disc geometry fsky=0.0216")
    # np.savetxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_SSC_disc_EE.txt",
    #            cov["KiDS-1000 SSC disc"]["EEEE"], header=header)

    # header = misc_utils.file_header(
    #                     f"Covariance EEEE SSC, mask power spectrum data/xcorr/cov/W_l/shear_KiDS1000_binary_auto.txt")
    # np.savetxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_SSC_mask_EE.txt",
    #            cov["KiDS-1000 SSC mask"]["EEEE"], header=header)

    # cov["KiDS-1000 m"] = cov_calculator.compute_m_covariance(
    #                             theory_cl_path=os.path.join(prediction_path, "shear_cl_binned"),
    #                             m=[0.019, 0.020, 0.017, 0.012, 0.010])
    # header = misc_utils.file_header(
    #                     f"Covariance EEEE m_correction, sigma_m=[0.019, 0.020, 0.017, 0.012, 0.010]")
    # np.savetxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_m_EE.txt",
    #            cov["KiDS-1000 m"]["EEEE"], header=header)

    # # Compute total
    # G = np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_gaussian_EE.txt")
    # SSC = np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_SSC_mask_EE.txt")
    # M = np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_m_EE.txt")

    # header = misc_utils.file_header(
    #                     f"Covariance EEEE Gaussian + SSC (mask) + m-correction")
    # np.savetxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_gaussian_SSC_mask_m_EE.txt",
    #            G+SSC+M, header=header)