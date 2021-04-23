import copy

import numpy as np
import scipy.interpolate

import pyccl as ccl

import pyhmcode

import misc_utils


PI = np.pi

import os
import sys
KCAP_PATH = "../../KiDS/kcap/"
sys.path.append(os.path.join(KCAP_PATH, "kcap"))
import cosmosis_utils


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
    def __init__(self, lensing_stats, y_stats, data_block,
                 cov_binning_operator):
        self.noise_spectra = {"shear": {}, "y": {}}

        self.data_block = data_block
        self._has_halo_profiles = False

        # Create shear spectra
        for shear_bin, s in enumerate(lensing_stats["bins"]):
            def shear_noise_spectrum(ell,
                                     sigma_e=s["sigma_e"], n_eff=s["n_eff"]):
                return np.ones_like(ell)*sigma_e**2/(n_eff*60**2/(1/180*PI)**2)

            self.noise_spectra["shear"][str(shear_bin+1)] \
                = shear_noise_spectrum

        # Create tSZ noise spectrum
        if "noise_file" in y_stats:
            self.noise_spectra["y"] = \
                self.create_tSZ_noise_spectrum_from_measured_noise(
                            noise_file=y_stats["noise_file"],
                            y_filter_fwhm=y_stats.get("y_filter_fwhm", None))

        elif "noise_model" in y_stats:
            self.noise_spectra["y"] = y_stats["noise_model"]

        self.f_sky = np.sqrt(lensing_stats["f_sky"]*y_stats["f_sky"])

        def section_suffix_mapping(probe_A, probe_B):
            if "y" in (probe_A, probe_B):
                return "_cl_beam_pixwin"
            else:
                return ""

        self.ell = np.arange(cov_binning_operator.shape[1])
        # Compute Gaussian covariances
        for name, noise_kwargs in [("gaussian",
                                    dict(noise=self.noise_spectra,
                                         noise_only=False)),
                                   ("gaussian_noise_only",
                                    dict(noise=self.noise_spectra,
                                         noise_only=True)),
                                   ("gaussian_no_noise",
                                    dict(noise=None, noise_only=False))]:
            cov = self.compute_gaussian_covariance(
                                probes=[(("shear", str(b)), "y")
                                        for b in self.noise_spectra["shear"]],
                                data=data_block,
                                ell=self.ell, f_sky=self.f_sky,
                                section_suffix_mapping=section_suffix_mapping,
                                binning_operator=cov_binning_operator,
                                **noise_kwargs)

            cov = np.vstack([np.hstack([np.diag(c) for c in c_i])
                             for c_i in cov])

            setattr(self, name, cov)

    def create_tSZ_noise_spectrum_from_measured_noise(self, noise_file,
                                                      y_filter_fwhm=None):
        ell_tSZ_raw, Cl_tSZ_raw_noise_homo = np.loadtxt(noise_file,
                                                        unpack=True)

        if y_filter_fwhm is not None:
            sigma = y_filter_fwhm/60.0/180.0*PI/(2.0*np.sqrt(2.0*np.log(2.0)))
            Cl_tSZ_raw_noise_homo *= np.exp(-0.5*ell_tSZ_raw**2*sigma**2)**2

        intp = scipy.interpolate.InterpolatedUnivariateSpline(
                    ell_tSZ_raw[2:], Cl_tSZ_raw_noise_homo[2:], ext=2)

        return intp

    def get_cl(self, block, suffix, probe_A, probe_B, bin_A, bin_B, ell,
               noise=None, noise_only=False):

        noise_spectrum = np.zeros(len(ell))
        if probe_A == probe_B and bin_A == bin_B:
            if (noise is not None
                    and probe_A in noise
                    and bin_A in noise[probe_A]):
                noise_spectrum = noise[probe_A][bin_A](ell)

        if noise_only:
            return noise_spectrum

        if probe_A == "shear" and probe_B == "shear":
            section = "shear_cl"
            key = f"bin_{bin_A}_{bin_B}"
            if not block.has_value(section, key):
                key = f"bin_{bin_B}_{bin_A}"
        else:
            section = f"{probe_A}_{probe_B}{suffix}"
            key = f"bin_{bin_A}_{bin_B}"
            if section not in block:
                section = f"{probe_B}_{probe_A}{suffix}"
                key = f"bin_{bin_B}_{bin_A}"

        cl_raw = block[section, key]
        ell_raw = block[section, "ell"]

        if np.all(cl_raw > 0):
            intp = scipy.interpolate.InterpolatedUnivariateSpline(
                        ell_raw, np.log(cl_raw), ext=2)
            cl = np.exp(intp(ell))
        else:
            intp = scipy.interpolate.InterpolatedUnivariateSpline(
                        ell_raw, cl_raw, ext=2)
            cl = intp(ell)

        return cl + noise_spectrum

    def compute_gaussian_covariance(self, probes, data, ell,
                                    noise=None, f_sky=1,
                                    section_suffix_mapping=None,
                                    binning_operator=None, noise_only=False):
        # Covariance between AB and CD

        suffix_mapping = section_suffix_mapping
        if suffix_mapping is None:
            def suffix_mapping(probe_A, probe_B):
                return "_cl"

        pad = np.count_nonzero(ell < 2)
        ell = ell[pad:]

        probes = [*probes]
        for i, (probe_A, probe_B) in enumerate(probes):
            probe_A = (probe_A, "1") if isinstance(probe_A, str) else probe_A
            probe_B = (probe_B, "1") if isinstance(probe_B, str) else probe_B
            probes[i] = (probe_A, probe_B)

        if noise is not None:
            noise = {**noise}
            for key, value in noise.items():
                if not isinstance(value, dict):
                    noise[key] = {"1": value}

        cov_list = []
        for i, ((probe_A, bin_A), (probe_B, bin_B)) in enumerate(probes):
            cov_list.append([])
            for j, ((probe_C, bin_C), (probe_D, bin_D)) in enumerate(probes):
                if j < i:
                    C = cov_list[j][i]
                else:
                    Cl_AC = self.get_cl(data, suffix_mapping(probe_A, probe_C),
                                        probe_A, probe_C, bin_A, bin_C, ell,
                                        noise, noise_only)
                    Cl_BD = self.get_cl(data, suffix_mapping(probe_B, probe_D),
                                        probe_B, probe_D, bin_B, bin_D, ell,
                                        noise, noise_only)
                    Cl_AD = self.get_cl(data, suffix_mapping(probe_A, probe_D),
                                        probe_A, probe_D, bin_A, bin_D, ell,
                                        noise, noise_only)
                    Cl_BC = self.get_cl(data, suffix_mapping(probe_B, probe_C),
                                        probe_B, probe_C, bin_B, bin_C, ell,
                                        noise, noise_only)

                    C = 1/((2*ell+1)*f_sky)*(Cl_AC*Cl_BD + Cl_AD*Cl_BC)

                    C = np.concatenate((np.zeros(pad), C))
                    if binning_operator is not None:
                        C = binning_operator @ C
                cov_list[i].append(C)

        return cov_list

    def compute_non_gaussian_covariance(self,
                                        Cl_binning_operator,
                                        beam_fwhm=None,
                                        n_ell_interpolate=100,
                                        cNG_term=True, verbose=False):
        data_block = self.data_block
        self.ccl_cosmo = ccl.Cosmology(
                    Omega_c=data_block["cosmological_parameters", "omega_c"],
                    Omega_b=data_block["cosmological_parameters", "omega_b"],
                    A_s=data_block["cosmological_parameters", "a_s"],
                    n_s=data_block["cosmological_parameters", "n_s"],
                    h=data_block["cosmological_parameters", "h0"],
                    m_nu=data_block["cosmological_parameters", "mnu"],
                                      )

        if not self._has_halo_profiles:
            a_arr = np.linspace(1/(1+6), 1, 32)
            k_arr = np.logspace(-5, 2, 100)
            self.compute_halo_profiles(data_block, a_arr, k_arr,
                                       verbose=verbose)

        if cNG_term:
            self.cNG = self.compute_cNG_covariance(data_block,
                                                   fsky_eff,
                                                   Cl_binning_operator,
                                                   beam_fwhm,
                                                   n_ell_interpolate,
                                                   verbose)

    def compute_halo_profiles(self, data_block, a_arr, k_arr, verbose=True):
        hmcode_cosmo = pyhmcode.Cosmology()

        hmcode_cosmo.om_m = data_block["cosmological_parameters", "omega_m"]
        hmcode_cosmo.om_b = data_block["cosmological_parameters", "omega_b"]
        hmcode_cosmo.om_v = \
            data_block["cosmological_parameters", "omega_lambda"]
        hmcode_cosmo.h = data_block["cosmological_parameters", "h0"]
        hmcode_cosmo.ns = data_block["cosmological_parameters", "n_s"]
        hmcode_cosmo.sig8 = data_block["cosmological_parameters", "sigma_8"]
        hmcode_cosmo.m_nu = data_block["cosmological_parameters", "mnu"]

        hmcode_cosmo.set_linear_power_spectrum(
                                    data_block["matter_power_lin", "k_h"],
                                    data_block["matter_power_lin", "z"],
                                    data_block["matter_power_lin", "p_k"])

        self.halo_profiles = HMxProfileGenerator(hmcode_cosmo,
                                                 a_arr=a_arr, k_arr=k_arr,
                                                 verbose=verbose)

        self._has_halo_profiles = True

    def compute_cNG_covariance(self, data_block,
                               fsky_eff,
                               binning_operator,
                               beam_fwhm=None,
                               n_ell_interpolate=100, verbose=True):
        if not self._has_halo_profiles:
            raise RuntimeError("Halo profiles not computed yet.")

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

        z_max = data_block["nz_kids1000", "z"][-1]

        a = np.linspace(1/(1+z_max), 1.0, 32)
        k = np.logspace(-4, 1.5, 100)

        if verbose:
            print("Computing halo model trispectrum")
        # Probably should switch the middle matter/pressure pair
        tk3D = ccl.halos.halomod_Tk3D_1h(
                            cosmo=self.ccl_cosmo, hmc=hmc,
                            prof1=self.halo_profiles.matter_profile,
                            prof2=self.halo_profiles.matter_profile,
                            prof12_2pt=None,
                            prof3=self.halo_profiles.pressure_profile,
                            prof4=self.halo_profiles.pressure_profile,
                            prof34_2pt=None,
                            lk_arr=np.log(k), a_arr=a,
                            use_log=True)

        ell = self.ell
        ell_coarse = np.linspace(2, ell[-1], n_ell_interpolate)

        n_z_bin = data_block["nz_kids1000", "nbin"]

        WL_tracers = [ccl.WeakLensingTracer(
                            self.ccl_cosmo,
                            dndz=(data_block["nz_kids1000", "z"],
                                  data_block["nz_kids1000", f"bin_{WL_bin}"]))
                      for WL_bin in range(1, n_z_bin+1)]

        tSZ_tracer = ccl.tSZTracer(self.ccl_cosmo, z_max=z_max)

        if beam_fwhm is not None:
            beam_operator = misc_utils.create_beam_operator(ell,
                                                            fwhm=beam_fwhm)
        else:
            beam_operator = None

        n_ell_bin = binning_operator.shape[0]

        cov_NG = np.zeros((len(WL_tracers), len(WL_tracers),
                           n_ell_bin, n_ell_bin))

        if verbose:
            print("Computing Cl cNG covariance term")
        for i, WL_tracer1 in enumerate(WL_tracers):
            for j, WL_tracer2 in enumerate(WL_tracers[:i+1]):
                cov_coarse = ccl.angular_cl_cov_cNG(
                                    cosmo=self.ccl_cosmo,
                                    cltracer1=WL_tracer1, cltracer2=WL_tracer2,
                                    cltracer3=tSZ_tracer, cltracer4=tSZ_tracer,
                                    ell=ell_coarse,
                                    tkka=tk3D, fsky=fsky_eff)
                cov = np.exp(scipy.interpolate.RectBivariateSpline(
                                                ell_coarse, ell_coarse,
                                                np.log(cov_coarse))(ell, ell))
                if beam_operator is not None:
                    cov = beam_operator @ cov @ beam_operator

                cov_NG[i, j] = binning_operator @ cov @ binning_operator.T

        cov_NG_matrix = np.zeros((len(WL_tracers)*n_ell_bin,
                                  len(WL_tracers)*n_ell_bin))
        for i in range(cov_NG.shape[0]):
            for j in range(cov_NG.shape[0])[: i+1]:
                cov_NG_matrix[i*n_ell_bin:(i+1)*n_ell_bin,
                              j*n_ell_bin:(j+1)*n_ell_bin] = cov_NG[i, j]
                if i != j:
                    cov_NG_matrix[j*n_ell_bin:(j+1)*n_ell_bin,
                                  i*n_ell_bin:(i+1)*n_ell_bin] = cov_NG[i, j]

        return cov_NG_matrix


def Planck_prediction(config, defaults):
    config_update = {"projection": {"y-y": "y-y"}}

    config.reset_config()
    config.update_config(config_update)
    block = config.run_pipeline(defaults=defaults)
    return block


def ACT_prediction(config, defaults):
    config_update = {"projection": {"y-y": "y-y"},
                     "beam_filter_cls": {"fwhm": 1.6}}

    config.reset_config()
    config.update_config(config_update)
    block = config.run_pipeline(defaults=defaults)
    return block


def load_Planck_random_shear_data(binning_operator, random_shear_data_path):
    random_shear_data = np.load(random_shear_data_path)

    y_map = "milca_half_difference"
    d = []
    z_cuts = [(0.1, 0.3),
              (0.3, 0.5),
              (0.5, 0.7),
              (0.7, 0.9),
              (0.9, 1.2),
              ]
    for z_cut in z_cuts:
        tag = f"z{z_cut[0]:.1f}-{z_cut[1]:.1f}-{y_map}"
        binned = np.einsum("ij,kjl->lik",
                           binning_operator,
                           random_shear_data[tag])
        d.append(binned)

    d = np.concatenate(d, axis=1)
    cov_random_shear = np.cov(d[0])

    return cov_random_shear


def load_ACT_random_shear_data():
    import glob

    random_shear_data_TE = []
    random_shear_data_TB = []

    for filename in glob.glob("../results/measurements/shear_KiDS1000_random_shear_y_ACT_BN_namaster/run_*/Cl_decoupled.txt"):
        d_TE = np.concatenate(np.loadtxt(filename, unpack=True, usecols=[1, 3, 5, 7, 9]))
        d_TB = np.concatenate(np.loadtxt(filename, unpack=True, usecols=[2, 4, 6, 8, 10]))
        random_shear_data_TE.append(d_TE)
        random_shear_data_TB.append(d_TB)

    # cov_random_shear_TE = np.cov(np.array(random_shear_data_TE).T, ddof=1)
    # cov_random_shear_TB = np.cov(np.array(random_shear_data_TB).T, ddof=1)
    cov_random_shear_combined = np.cov(np.array(random_shear_data_TE+random_shear_data_TB).T, ddof=1)

    return cov_random_shear_combined


def ACT_BN_noise_model(auto_power_spectrum_file, data_block):
    ell = data_block["y_y_cl_beam_pixwin", "ell"]
    cl = data_block["y_y_cl_beam_pixwin", "bin_1_1"]

    cl_signal_intp = scipy.interpolate.InterpolatedUnivariateSpline(ell, cl)

    ell_data, cl_data = np.loadtxt(auto_power_spectrum_file, unpack=True)
    cl_noise = cl_data - cl_signal_intp(ell_data)

    cl_noise_intp = scipy.interpolate.InterpolatedUnivariateSpline(
                        ell_data, cl_noise)

    return cl_noise_intp


def calibrate_xcorr_area(binning_operator, binning_operator_squared,
                         data_block,
                         KiDS_stats, CMB_stats,
                         random_shear_cov):
    print()
    print("Finding effective area for cross-correlation")
    print()

    lensing_stats = copy.copy(KiDS1000_stats)
    y_stats = copy.copy(CMB_stats)

    fsky_initial = KiDS_stats["f_sky"]

    lensing_stats["f_sky"] = fsky_initial
    y_stats["f_sky"] = fsky_initial

    covariance = CovarianceCalculator(
                            lensing_stats=lensing_stats,
                            y_stats=y_stats,
                            data_block=data_block,
                            cov_binning_operator=binning_operator_squared)

    noise_cov_initial = covariance.gaussian_noise_only

    def noise_covariance_loss(fsky_eff):
        scaled_cov = (fsky_initial/fsky_eff)*noise_cov_initial

        return np.sum((np.diag(random_shear_cov)*1e28
                       - np.diag(scaled_cov)*1e28)**2)

    print("Fitting fsky_eff")
    result = scipy.optimize.minimize(noise_covariance_loss, x0=fsky_initial)

    fsky_eff = result.x[0]
    print("A_eff", fsky_eff*4*PI*(180/PI)**2)

    return (fsky_eff,
            (fsky_initial/fsky_eff)*noise_cov_initial)


def plot_noise_cov(binning_operator, covs, title="", filename=""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8, 5))
    fig.subplots_adjust(hspace=0, wspace=0)

    n_ell_bin = binning_operator.shape[0]

    ell = np.arange(binning_operator.shape[1])
    ell_eff = binning_operator @ ell
    u = ell_eff**3

    z_cuts = [(0.1, 0.3),
              (0.3, 0.5),
              (0.5, 0.7),
              (0.7, 0.9),
              (0.9, 1.2),
              ]
    for i, z_cut in enumerate(z_cuts):
        for cov_spec in covs:
            c = cov_spec.copy()
            cov = c.pop("cov")
            var = np.diag(cov)[n_ell_bin*i:n_ell_bin*(i+1)]

            ax.flatten()[i].loglog(ell_eff, u*var, **c)

        ax.flatten()[i].set_xlabel(r"$\ell$")
        ax.flatten()[i].set_title(f"z: {z_cut[0]}-{z_cut[1]}", x=0.25, y=0.85)

    [p[0].set_ylabel(r"$\ell^3\ {\rm Var}[C_\ell]$") for p in ax]
    ax.flatten()[-1].axis("off")
    ax.flatten()[-2].legend(frameon=False, loc="upper left",
                            bbox_to_anchor=(1, 0.8))

    ax[0,0].set_ylim(bottom=1e-22)

    fig.suptitle(title)
    fig.dpi = 300
    if filename != "":
        fig.savefig(filename)


if __name__ == "__main__":
    import base_config
    import tSZ_pipeline

    defaults = {**base_config.PATHS, **base_config.DATA_DIRS}

    binning_operator = np.loadtxt(
                            cosmosis_utils.emulate_configparser_interpolation(
                                base_config.bin_op_file, defaults))

    binning_operator_squared = np.loadtxt(
                            cosmosis_utils.emulate_configparser_interpolation(
                                base_config.bin_op_squared_file, defaults))

    KiDS1000_f_sky = 773.3/(4*PI*(180/PI)**2)

    A_eff = 636.8005428
    fsky_eff = A_eff/(4*PI*(180/PI)**2)

    KiDS1000_stats = {"bins": [{"n_eff": 0.62, "sigma_e": 0.27},
                               {"n_eff": 1.18, "sigma_e": 0.26},
                               {"n_eff": 1.85, "sigma_e": 0.27},
                               {"n_eff": 1.26, "sigma_e": 0.25},
                               {"n_eff": 1.31, "sigma_e": 0.27}],
                      "f_sky": fsky_eff}

    KiDS1000_N_stats = {"bins": [{"n_eff": 0.62, "sigma_e": 0.27},
                                 {"n_eff": 1.20, "sigma_e": 0.26},
                                 {"n_eff": 1.82, "sigma_e": 0.27},
                                 {"n_eff": 1.19, "sigma_e": 0.25},
                                 {"n_eff": 1.17, "sigma_e": 0.27}],
                        "f_sky": fsky_eff}

    OLD_MEASUREMENT_DIR = "../../project-triad-obsolete/results/measurements/"

    Planck_milca_stats = {"noise_file": os.path.join(OLD_MEASUREMENT_DIR,
                            "y_y/milca-half-difference-ellmax_4000/spice.cl"),
                          "f_sky": fsky_eff}


    config = tSZ_pipeline.tSZPipelineFactory(
                options=dict(
                            source_nz_files=base_config.nz_files,
                            dz_covariance_file=base_config.nz_cov_file,
                            source_nz_sample="KiDS1000",
                            y_filter_fwhm=10.0, nside=2048,
                            ))

    blocks = {}
    
    compute_ACT_BN_cov = True
    compute_Planck_milca_cov = True

    if compute_ACT_BN_cov:
        print("Computing covariance for ACT BN")
        print()

        blocks["cov_ACT"] = ACT_prediction(config, defaults=defaults)

        ACT_BN_noise_stats = {"noise_file": "../results/measurements/y_y_ACT_BN_namaster/Cl_TT_decoupled_unbinned.txt",
                              "f_sky": fsky_eff}

        random_shear_ACT_BN_cov = load_ACT_random_shear_data()

        fsky_eff_ACT_BN, noise_cov_ACT_BN = \
            calibrate_xcorr_area(binning_operator, binning_operator_squared,
                                 blocks["cov_ACT"],
                                 KiDS1000_N_stats, ACT_BN_noise_stats,
                                 random_shear_ACT_BN_cov)

        A_eff_ACT_BN = fsky_eff_ACT_BN*4*PI*(180/PI)**2

        KiDS1000_N_stats["f_sky"] = fsky_eff_ACT_BN
        ACT_BN_stats = {"noise_model": ACT_BN_noise_model("../results/measurements/y_y_ACT_BN_namaster/Cl_TT_decoupled_unbinned.txt",
                                                      blocks["cov_ACT"]),
                        "f_sky": fsky_eff_ACT_BN}

        covariance = CovarianceCalculator(
                            lensing_stats=KiDS1000_N_stats,
                            y_stats=ACT_BN_stats,
                            data_block=blocks["cov_ACT"],
                            cov_binning_operator=binning_operator_squared)

        covariance.compute_non_gaussian_covariance(
                                        Cl_binning_operator=binning_operator,
                                        beam_fwhm=1.6,
                                        n_ell_interpolate=100,
                                        verbose=False)

        header = misc_utils.file_header(
                                f"A_eff = {A_eff_ACT_BN} sq. deg., "
                                f"binning scheme: {base_config.bin_op_file}")
        y_map = "ACT_BN"
        for name, cov in [("gaussian", covariance.gaussian),
                          ("gaussian_cNG", covariance.gaussian+covariance.cNG)]:
            np.savetxt(f"../data/xcorr/cov/shear_y_KiDS1000_N_{y_map}_TE_{name}_cov.txt", 
                       cov, header=header)

        plot_noise_cov(binning_operator,
                    covs=[
                        {"cov":   random_shear_ACT_BN_cov,
                        "label": "Random shear x ACT BN y map",
                        "ls":    "-"},
                        {"cov":   noise_cov_ACT_BN,
                        "label":  "Noise covariance",
                        "ls":     "--"}],
                    title=f"ACT BN y x KiDS-1000-N noise covariance (A_eff = {A_eff_ACT_BN:.1f} deg$^2$)",
                    filename="../notebooks/plots/ACT_BN_noise_cov.png")

        plot_noise_cov(binning_operator,
                    covs=[
                        {"cov":   random_shear_ACT_BN_cov,
                         "label": "Random shear x ACT BN y map",
                         "ls":    "-"},
                        {"cov":   noise_cov_ACT_BN,
                         "label":  "Shear noise covariance",
                         "ls":     ":"},
                        {"cov":   covariance.gaussian,
                         "label":  "Gaussian covariance",
                         "ls":     "-"},
                        {"cov":   covariance.gaussian_no_noise,
                         "label":  "Gaussian sample covariance",
                         "ls":     ":"},
                        {"cov":   covariance.gaussian_noise_only,
                         "label":  "Gaussian noise covariance",
                         "ls":     ":"},
                        {"cov":   covariance.cNG,
                         "label":  "cNG",
                         "ls":     ":"},
                        {"cov":   covariance.gaussian+covariance.cNG,
                         "label":  "Total",
                         "ls":     "-", "lw": 2},],
                    title=f"ACT BN y x KiDS-1000-N covariance (A_eff = {A_eff_ACT_BN:.1f} deg$^2$)",
                    filename="../notebooks/plots/ACT_BN_cov.png")


    if compute_Planck_milca_cov:
        print("Computing covariance for Planck milca")
        print()

        blocks["cov_Planck"] = Planck_prediction(config, defaults=defaults)

        Planck_random_shear_data_path = os.path.join(
                                            OLD_MEASUREMENT_DIR,
                                            "shear_KiDS1000_y/"
                                            "shear_KiDS1000_y_random_shear.npz")
        random_shear_Planck_cov = load_Planck_random_shear_data(
                                        binning_operator,
                                        Planck_random_shear_data_path)

        fsky_eff_Planck, noise_cov_Planck = \
            calibrate_xcorr_area(binning_operator, binning_operator_squared,
                                 blocks["cov_Planck"],
                                 KiDS1000_stats, Planck_milca_stats,
                                 random_shear_Planck_cov)

        A_eff_Planck = fsky_eff_Planck*4*PI*(180/PI)**2

        KiDS1000_stats["f_sky"] = fsky_eff_Planck
        Planck_milca_stats["f_sky"] = fsky_eff_Planck

        covariance = CovarianceCalculator(
                            lensing_stats=KiDS1000_stats,
                            y_stats=Planck_milca_stats,
                            data_block=blocks["cov_Planck"],
                            cov_binning_operator=binning_operator_squared)

        covariance.compute_non_gaussian_covariance(
                                        Cl_binning_operator=binning_operator,
                                        beam_fwhm=10.0,
                                        n_ell_interpolate=100,
                                        verbose=False)

        header = misc_utils.file_header(
                                f"A_eff = {A_eff_Planck} sq. deg., "
                                f"binning scheme: {base_config.bin_op_file}")
        y_map = "milca"
        for name, cov in [("gaussian", covariance.gaussian),
                          ("gaussian_cNG", covariance.gaussian+covariance.cNG)]:
            np.savetxt(f"../data/xcorr/cov/shear_y_KiDS1000_{y_map}_TE_{name}_cov.txt", 
                       cov, header=header)

        plot_noise_cov(binning_operator,
                    covs=[
                        {"cov":   random_shear_Planck_cov,
                         "label": "Random shear \n  x Planck milca half-difference",
                         "ls":    "--"},
                        {"cov":   noise_cov_Planck,
                         "label":  "Noise covariance",
                         "ls":     ":"},
                        {"cov":   covariance.gaussian,
                         "label":  "Gaussian covariance",
                         "ls":     "-"},
                        {"cov":   covariance.gaussian_no_noise,
                         "label":  "Gaussian sample covariance",
                         "ls":     ":"},
                        {"cov":   covariance.gaussian_noise_only,
                         "label":  "Gaussian noise covariance",
                         "ls":     ":"},
                        {"cov":   covariance.cNG,
                         "label":  "cNG",
                         "ls":     ":"},
                        {"cov":   covariance.gaussian+covariance.cNG,
                         "label":  "Total",
                         "ls":     "-", "lw": 2},],
                    title=f"Planck milca x KiDS-1000 noise covariance (A_eff = {A_eff_Planck:.1f} deg$^2$)",
                    filename="../notebooks/plots/Planck_milca_cov.png")

        