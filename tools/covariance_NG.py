import os
import pickle

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


def ccl2hmcode_cosmo(ccl_cosmo, pofk_lin_z, pofk_lin_k_h, pofk_lin,
                     log10_T_heat=None):
    hmcode_cosmo = pyhmcode.Cosmology()

    hmcode_cosmo.om_m = ccl_cosmo["Omega_m"]
    hmcode_cosmo.om_b = ccl_cosmo["Omega_b"]
    hmcode_cosmo.om_v = ccl_cosmo["Omega_l"]
    hmcode_cosmo.h = ccl_cosmo["h"]
    hmcode_cosmo.ns = ccl_cosmo["n_s"]
    sigma8 = ccl_cosmo["sigma8"]
    if not np.isfinite(sigma8):
        sigma8 = ccl.sigma8(ccl_cosmo)
    hmcode_cosmo.sig8 = sigma8
    hmcode_cosmo.m_nu = ccl_cosmo["m_nu"].sum()

    if log10_T_heat is not None:
        hmcode_cosmo.theat = 10**log10_T_heat

    hmcode_cosmo.set_linear_power_spectrum(pofk_lin_k_h,
                                           pofk_lin_z,
                                           pofk_lin)

    return hmcode_cosmo


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
    def __init__(self, hmcode_cosmo, a_arr, k_arr, fields=None,
                 add_diffuse=False, verbose=False):
        self.hmcode_cosmo = hmcode_cosmo
        self.hmod = pyhmcode.Halomodel(
                        pyhmcode.HMx2020_matter_pressure_w_temp_scaling,
                        verbose=verbose)

        self.a_arr = a_arr.copy()
        self.k_arr = k_arr.copy()

        self.verbose = verbose

        if fields is None:
            self.fields = [pyhmcode.field_matter,
                           pyhmcode.field_electron_pressure]
        else:
            self.fields = fields

        self.add_diffuse = add_diffuse

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
                if self.add_diffuse:
                    pyhmcode.hmx.add_smooth_component_to_windows(
                                        self.fields, len(self.fields),
                                        wk, self.hmod.n,
                                        self.hmod, self.hmcode_cosmo)
                profile_lut[:, a_idx, k_idx] = wk.T

        if pyhmcode.field_electron_pressure in self.fields:
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


def make_matrix_symmetric(M):
    m = M.copy()
    m[np.triu_indices_from(M, k=1)] = m.T[np.triu_indices_from(M, k=1)]
    return m


class CovarianceCalculator:
    def __init__(self, ccl_cosmo, lensing_nz,
                 log10_T_heat=None, pofk_lin=None):
        self.ccl_cosmo = ccl_cosmo
        self.log10_T_heat = log10_T_heat

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

    def compute_halo_profiles(self, a_arr, k_arr, diffuse=False, verbose=True):
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

        hmcode_cosmo.theat = 10**self.log10_T_heat

        hmcode_cosmo.set_linear_power_spectrum(self.pofk_lin_k_h,
                                               self.pofk_lin_z,
                                               self.pofk_lin)

        self.halo_profiles = HMxProfileGenerator(
                                        hmcode_cosmo,
                                        a_arr=a_arr,
                                        k_arr=k_arr/self.ccl_cosmo["h"],
                                        verbose=verbose)
        if diffuse:
            self.halo_profiles_diffuse = HMxProfileGenerator(
                                        hmcode_cosmo,
                                        a_arr=a_arr,
                                        k_arr=k_arr/self.ccl_cosmo["h"],
                                        add_diffuse=True,
                                        verbose=verbose)
        else:
            self.halo_profiles_diffuse = None

        self._has_halo_profiles = True

    def compute_trispectra(self, probes, a_arr, k_arr,
                           pofk_background_linear=None,
                           cNG=True, SSC=True,
                           halo_model="hmx", verbose=True):

        profiles = {}
        profiles_2h = {}
        if halo_model == "hmx":
            if not self._has_halo_profiles:
                self.compute_halo_profiles(a_arr, k_arr, diffuse=SSC,
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
            if SSC:
                profiles_2h["matter"] = \
                    self.halo_profiles_diffuse.matter_profile
                profiles_2h["pressure"] = \
                    self.halo_profiles_diffuse.pressure_profile
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
            profiles_2h["matter"] = profiles["matter"]
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
                                    prof1=profiles_2h[a1],
                                    prof2=profiles_2h[a2],
                                    prof12_2pt=None,
                                    prof3=profiles_2h[b1],
                                    prof4=profiles_2h[b2],
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

        # Turn fsky/mask_wl into dict for the requested cov_blocks
        if fsky is not None and not isinstance(fsky, dict):
            fsky = {c: fsky for c in cov_blocks}
        if mask_wl is not None and not isinstance(mask_wl, dict):
            mask_wl = {c: mask_wl for c in cov_blocks}

        if mode == "SSC":
            sigma2_B = {}
            if mask_wl is not None:
                for cov_block, m in mask_wl.items():
                    a = a_arr.copy()
                    s = (a, [ccl.covariances.sigma2_B_from_mask(
                                                    self.ccl_cosmo,
                                                    a=a_,
                                                    mask_wl=m)
                             for a_ in a])
                    sigma2_B[cov_block] = s
                fsky = {c: None for c in cov_blocks}
            else:
                for cov_block, f in fsky.items():
                    sigma2_B[cov_block] = (a_arr,
                                           ccl.covariances.sigma2_B_disc(
                                                        self.ccl_cosmo,
                                                        a=a_arr, fsky=f))
        else:
            sigma2_B = {c: None for c in cov_blocks}

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
                                    fsky=fsky[cov_block],
                                    sigma2_B=sigma2_B[cov_block],
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
                                    fsky=fsky[cov_block],
                                    sigma2_B=sigma2_B[cov_block],
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
                                    fsky=fsky[cov_block],
                                    sigma2_B=sigma2_B[cov_block],
                                    beam_operator_a=None,
                                    beam_operator_b=beam_operator,
                                    n_ell_intp=n_ell_intp)
                        cov["EETE"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                    j*n_ell_bin_TE:(j+1)*n_ell_bin_TE] = c.T

        if "joint" in cov_blocks:
            cov["joint"] = self.assemble_joint_covariance(cov)

        for cov_block in cov_blocks:
            cov[cov_block] = make_matrix_symmetric(cov[cov_block])

        return cov

    def compute_m_covariance(self, theory_cl_paths, m, cov_blocks):
        # See eq. 37 in Joachimi+2021 (2007.01844)
        field_idx_EE = [(i, j) for i in range(self.n_z)
                        for j in range(i+1)]
        field_idx_TE = [(i, 0) for i in range(self.n_z)]

        if "EEEE" in cov_blocks or "EETE" in cov_blocks:
            Cl_EE = []
            for i, j in field_idx_EE:
                d = np.loadtxt(
                        os.path.join(theory_cl_paths["EE"],
                                     f"bin_{i+1}_{j+1}.txt"))
                Cl_EE.append(d)
            Cl_EE = np.concatenate(Cl_EE)

        if "TETE" in cov_blocks or "EETE" in cov_blocks:
            Cl_TE = []
            for i, j in field_idx_TE:
                d = np.loadtxt(
                        os.path.join(theory_cl_paths["TE"],
                                     f"bin_{i+1}_{j+1}.txt"))
                Cl_TE.append(d)
            Cl_TE = np.concatenate(Cl_TE)

        cov = {}
        for cov_block in cov_blocks:
            if cov_block == "EEEE":
                field_idx_a, field_idx_b = field_idx_EE, field_idx_EE
                Cl_a, Cl_b = Cl_EE, Cl_EE
                m_a1, m_a2 = m, m
                m_b1, m_b2 = m, m
            if cov_block == "TETE":
                field_idx_a, field_idx_b = field_idx_TE, field_idx_TE
                Cl_a, Cl_b = Cl_TE, Cl_TE
                m_a1, m_a2 = m, np.zeros_like(m)
                m_b1, m_b2 = m, np.zeros_like(m)
            if cov_block == "EETE":
                field_idx_a, field_idx_b = field_idx_EE, field_idx_TE
                Cl_a, Cl_b = Cl_EE, Cl_TE
                m_a1, m_a2 = m, m
                m_b1, m_b2 = m, np.zeros_like(m)

            Cl_matrix = np.outer(Cl_a, Cl_b)
            n_ell = d.size
            m_matrix = np.zeros_like(Cl_matrix)
            for i, (idx_a1, idx_a2) in enumerate(field_idx_a):
                for j, (idx_b1, idx_b2) in enumerate(field_idx_b):
                    m_matrix[i*n_ell: (i+1)*n_ell, j*n_ell: (j+1)*n_ell] = (
                                        m_a1[idx_a1]*m_b1[idx_b1]
                                        + m_a1[idx_a1]*m_b2[idx_b2]
                                        + m_a2[idx_a2]*m_b1[idx_b1]
                                        + m_a2[idx_a2]*m_b2[idx_b2])

            cov[cov_block] = m_matrix*Cl_matrix

        if {"EEEE", "TETE", "EETE"} <= cov.keys():
            cov["joint"] = self.assemble_joint_covariance(cov)

        return cov

    def assemble_joint_covariance(self, covs):
        n_EE = covs["EEEE"].shape[0]
        n_TE = covs["TETE"].shape[0]

        cov = np.zeros((n_EE+n_TE, n_EE+n_TE))
        cov[:n_EE, :n_EE] = covs["EEEE"]
        cov[n_EE:n_EE+n_TE, :n_EE] = covs["EETE"].T
        cov[n_EE:n_EE+n_TE, n_EE:n_EE+n_TE] = covs["TETE"]

        cov = make_matrix_symmetric(cov)

        return cov


if __name__ == "__main__":

    # base_path = "../results/measurements/shear_KiDS1000_y_milca/"
    base_path = "../results/measurements/shear_KiDS1000_cel_y_ACT_BN/"

    # beam = 10.0
    beam = 1.6

    # b = np.loadtxt("../data/xcorr/bin_operator_log_n_bin_12_ell_51-2952.txt")
    # binning_operator = {"EE": b, "TE": b}

    binning_operator = {"EE": np.load("../results/measurements/shear_KiDS1000_shear_KiDS1000/data/pymaster_bandpower_windows_4-4.npy")[0, :, 0],  # noqa: E501
                        # "TE": np.load("../results/measurements/shear_KiDS1000_y_milca/data/pymaster_bandpower_windows_4-0.npy")[0, :, 0]}         # noqa: E501
                        "TE": np.load("../results/measurements/shear_KiDS1000_cel_y_ACT_BN/data/pymaster_bandpower_windows_4-0.npy")[0, :, 0]}         # noqa: E501

    # From data/xcorr/cov/W_l/
    # in sr
    footprint_areas = {"KiDS1000":  0.2714,
                       "Planck_gal40_ps": 6.2120,
                       "KiDS1000_Planck_gal40_ps_overlap": 0.2323,
                       "ACT_BN_Planck_gal40_ps": 0.4403,
                       "KiDS1000_cel_ACT_BN_Planck_gal40_ps_overlap": 0.0825}

    fsky_EE = footprint_areas["KiDS1000"]/(4*np.pi)
    # fsky_TE = footprint_areas["KiDS1000_Planck_gal40_ps_overlap"]/(4*np.pi)
    fsky_TE = footprint_areas["KiDS1000_cel_ACT_BN_Planck_gal40_ps_overlap"]/(4*np.pi)

    fsky = {"EEEE": fsky_EE,
            "TETE": fsky_TE,
            "EETE": np.sqrt(fsky_EE*fsky_TE)}

    # mask_wl_files = {"EEEE": "../data/xcorr/cov/W_l/shear_KiDS1000_binary_auto.txt",                                  # noqa: E501
    #                  "TETE": "../data/xcorr/cov/W_l/shear_KiDS1000_Planck_gal40_ps_binary.txt",                       # noqa: E501
    #                  "EETE": "../data/xcorr/cov/W_l/shear_KiDS1000_Planck_gal40_ps_binary.txt"}                       # noqa: E501

    mask_wl_files = {"EEEE": "../data/xcorr/cov/W_l/shear_KiDS1000_binary_auto.txt",                                  # noqa: E501
                     "TETE": "../data/xcorr/cov/W_l/shear_KiDS1000_cel_ACT_BN_Planck_gal40_ps_binary.txt",                       # noqa: E501
                     "EETE": "../data/xcorr/cov/W_l/shear_KiDS1000_cel_ACT_BN_Planck_gal40_ps_binary.txt"}                       # noqa: E501


    # mask_wl_overlap_files = {"EEEE": "../data/xcorr/cov/W_l/shear_KiDS1000_binary_auto.txt",                          # noqa: E501
    #                          "TETE": "../data/xcorr/cov/W_l/shear_KiDS1000_Planck_gal40_ps_overlap_binary_auto.txt",  # noqa: E501
    #                          "EETE": "../data/xcorr/cov/W_l/shear_KiDS1000_Planck_gal40_ps_overlap_binary_auto.txt"}  # noqa: E501

    mask_wl = {k: np.loadtxt(f, usecols=[1]) for k, f in mask_wl_files.items()}
    # mask_wl_overlap = {k: np.loadtxt(f, usecols=[1]) for k, f in mask_wl_overlap_files.items()}          # noqa: E501

    # prediction_path = "../data/xcorr/theory_predictions/cov_theory_predictions_run1_hmx_nz128_beam10/output/data_block/"        # noqa: E501
    prediction_path = "../data/xcorr/theory_predictions/cov_theory_predictions_run3_hmx_nocib_beam1.6/output/data_block/"        # noqa: E501

    cosmo_params = load_cosmosis_params(prediction_path)
    halo_model_params = load_cosmosis_params(prediction_path, "halo_model_parameters")                 # noqa: E501

    pofk_lin = np.loadtxt(os.path.join(prediction_path, "matter_power_lin/p_k.txt"))                   # noqa: E501
    pofk_lin_z = np.loadtxt(os.path.join(prediction_path, "matter_power_lin/z.txt"))                   # noqa: E501
    pofk_lin_k = np.loadtxt(os.path.join(prediction_path, "matter_power_lin/k_h.txt"))                 # noqa: E501

    pofk_lin = (pofk_lin, pofk_lin_k, pofk_lin_z)

    ccl_cosmo = ccl.Cosmology(Omega_c=cosmo_params["omega_c"],
                              Omega_b=cosmo_params["omega_b"],
                              Omega_k=cosmo_params["omega_k"],
                              n_s=cosmo_params["n_s"],
                              sigma8=cosmo_params["sigma_8"],
                              h=cosmo_params["h0"],
                              m_nu=cosmo_params["mnu"])

    ell_max = binning_operator["EE"].shape[1]

    ell = np.arange(ell_max)
    beam_operator = misc_utils.create_beam_operator(ell,
                                                    fwhm=beam,
                                                    )

    z = np.loadtxt(os.path.join(prediction_path, "nz_kids1000/z.txt"))
    lensing_nz = [(z, np.loadtxt(os.path.join(prediction_path, f"nz_kids1000/bin_{i+1}.txt")))      # noqa: E501
                  for i in range(5)]

    cov_calculator = CovarianceCalculator(
                            ccl_cosmo, lensing_nz,
                            pofk_lin=pofk_lin,
                            log10_T_heat=halo_model_params["log10_theat"])

    try:
        with open(os.path.join(base_path, "cov_NG/cov_NG.pickle"), "rb") as f:
            cov = pickle.load(f)
    except FileNotFoundError:
        cov = {}

    # cov["KiDS-1000xtSZ SSC disc"] = cov_calculator.compute_NG_covariance(
    #                         mode="SSC",
    #                         cov_blocks=["TETE", "EETE", "EEEE", "joint"],
    #                         binning_operator=binning_operator,
    #                         beam_operator=beam_operator,
    #                         fsky=fsky,
    #                         halo_model="hmx",
    #                         n_ell_intp=100)
    # with open(
    #         os.path.join(base_path, "cov_NG/cov_NG.pickle"), "wb") as f:
    #     pickle.dump(cov, f)

    cov["KiDS-1000xtSZ SSC mask"] = cov_calculator.compute_NG_covariance(
                            mode="SSC",
                            cov_blocks=["TETE", "EETE", "EEEE", "joint"],
                            binning_operator=binning_operator,
                            beam_operator=beam_operator,
                            mask_wl=mask_wl,
                            halo_model="hmx",
                            n_ell_intp=100)
    with open(
            os.path.join(base_path, "cov_NG/cov_NG.pickle"), "wb") as f:
        pickle.dump(cov, f)

    # cov["KiDS-1000xtSZ SSC mask overlap"] = \
    #     cov_calculator.compute_NG_covariance(
    #                         mode="SSC",
    #                         cov_blocks=["TETE", "EETE", "EEEE", "joint"],
    #                         binning_operator=binning_operator,
    #                         beam_operator=beam_operator,
    #                         mask_wl=mask_wl_overlap,
    #                         halo_model="hmx",
    #                         n_ell_intp=100)
    # with open(
    #         os.path.join(base_path, "cov_NG/cov_NG.pickle"), "wb") as f:
    #     pickle.dump(cov, f)

    cov["KiDS-1000xtSZ cNG"] = cov_calculator.compute_NG_covariance(
                            mode="cNG",
                            cov_blocks=["TETE", "EETE", "EEEE", "joint"],
                            binning_operator=binning_operator,
                            beam_operator=beam_operator,
                            fsky=fsky,
                            halo_model="hmx",
                            n_ell_intp=100)
    with open(
            os.path.join(base_path, "cov_NG/cov_NG.pickle"), "wb") as f:
        pickle.dump(cov, f)

    theory_cl_paths = {"EE": os.path.join(prediction_path,
                                          "shear_cl_binned"),
                       "TE": os.path.join(prediction_path,
                                          "shear_y_cl_beam_pixwin_binned")}
    cov["KiDS-1000xtSZ m"] = cov_calculator.compute_m_covariance(
                                theory_cl_paths=theory_cl_paths,
                                m=[0.019, 0.020, 0.017, 0.012, 0.010],
                                cov_blocks=["EEEE", "TETE", "EETE"])
    with open(
            os.path.join(base_path, "cov_NG/cov_NG.pickle"), "wb") as f:
        pickle.dump(cov, f)

    os.makedirs(os.path.join(base_path, "likelihood/cov"), exist_ok=True)

    for cov_block in ["EEEE", "TETE", "joint"]:
        if cov_block == "joint":
            fsky_str = f"{fsky}"
            mask_wl_str = f"{mask_wl_files}"
            # mask_wl_overlap_str = f"{mask_wl_overlap_files}"
        else:
            fsky_str = f"{fsky[cov_block]}"
            mask_wl_str = f"{mask_wl_files[cov_block]}"
            # mask_wl_overlap_str = f"{mask_wl_overlap_files[cov_block]}"

        header = misc_utils.file_header(
                    f"Covariance {cov_block} m_correction, sigma_m=[0.019, 0.020, 0.017, 0.012, 0.010]")                # noqa: E501
        np.savetxt(
            os.path.join(
                base_path,
                f"likelihood/cov/covariance_m_{cov_block}.txt"),
            cov["KiDS-1000xtSZ m"][cov_block], header=header)

        # header = misc_utils.file_header(
        #             f"Covariance HMx {cov_block} SSC, disc geometry fsky={fsky_str}")                                   # noqa: E501
        # np.savetxt(
        #     os.path.join(
        #         base_path,
        #         f"likelihood/cov/covariance_hmx_SSC_disc_{cov_block}.txt"),
        #     cov["KiDS-1000xtSZ SSC disc"][cov_block], header=header)

        header = misc_utils.file_header(
                    f"Covariance HMx {cov_block} SSC, mask. Mask power file: {mask_wl_str}")              # noqa: E501
        np.savetxt(
            os.path.join(
                base_path,
                f"likelihood/cov/covariance_hmx_SSC_mask_wl_{cov_block}.txt"),
            cov["KiDS-1000xtSZ SSC mask"][cov_block], header=header)

        # header = misc_utils.file_header(
        #             f"Covariance HMx {cov_block} SSC, mask, overlap only. Mask power file: {mask_wl_overlap_str}")      # noqa: E501
        # np.savetxt(
        #     os.path.join(
        #         base_path,
        #         f"likelihood/cov/"
        #         f"covariance_hmx_SSC_mask_wl_overlap_{cov_block}.txt"),
        #     cov["KiDS-1000xtSZ SSC mask overlap"][cov_block], header=header)

        header = misc_utils.file_header(
                    f"Covariance HMx EEEE cNG, 1h fsky={fsky_str}")
        np.savetxt(
            os.path.join(
                base_path,
                f"likelihood/cov/covariance_hmx_cNG_1h_{cov_block}.txt"),
            cov["KiDS-1000xtSZ cNG"][cov_block], header=header)
