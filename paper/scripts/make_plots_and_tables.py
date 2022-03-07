import copy
import os
import warnings
import glob
import argparse
import pickle

import matplotlib.pyplot as plt
import matplotlib

import getdist
import getdist.plots
import numpy as np
import scipy.stats

import camb

try:
    import anesthetic
    import tensiometer

    import tensiometer.gaussian_tension
    import tensiometer.mcmc_tension
except ImportError:
    print("Couldn't import anesthetic and/or tensionmeter")

KCAP_PATH = "../../../KiDS/kcap/"
import sys
sys.path.append(KCAP_PATH + "utils/")

import process_chains
import stat_tools
import hellinger_distance_1D

sys.path.append("../tools/")
import plotting_utils

pi = np.pi

import getdist.chains
getdist.chains.print_load_details = False


def get_MAP(MAP_path, verbose=False):
    MAP_max_logpost = -np.inf
    MAPs = {}
    files = {}
    MAP_files = []
    if not isinstance(MAP_path, (list, tuple)):
        MAP_path = [MAP_path]
    for path in MAP_path:
        MAP_files += glob.glob(path)
    if len(MAP_files) == 0:
        raise RuntimeError("No MAP sample files found.")
        
    for file in MAP_files:
        try:
            MAP_chain = process_chains.load_chain(file, run_name="MAP", burn_in=0, 
                                                  ignore_inf=True, strict_mapping=False)
        except RuntimeError:
            continue
            
        if "omegamh2" not in [n.name for n in MAP_chain.getParamNames().names]:
            MAP_chain.addDerived(MAP_chain.getParams().omegam*MAP_chain.getParams().h**2,
                         name="omegamh2", label="\\Omega_{\\rm m} h^2")
        if "omeganuh2" not in [n.name for n in MAP_chain.getParamNames().names]:
            MAP_chain.addDerived(MAP_chain.getParams().omeganu*MAP_chain.getParams().h**2,
                     name="omeganuh2", label="\\Omega_{\\nu} h^2")
        if "Sigmaalpha" not in [n.name for n in MAP_chain.getParamNames().names]:
            MAP_chain.addDerived(MAP_chain.getParams().sigma8*(MAP_chain.getParams().omegam/0.3)**0.58,
                         name="Sigmaalpha", label="\\Sigma_{8}")
        if "Sigma0p2" not in [n.name for n in MAP_chain.getParamNames().names]:
            MAP_chain.addDerived(MAP_chain.getParams().sigma8*(MAP_chain.getParams().omegam/0.3)**0.2,
                     name="Sigma0p2", label="\\Sigma_{8}^{\\alpha=0.2}")

        MAP_logpost = MAP_chain.getParams().logpost.max()
        MAP_idx = MAP_chain.getParams().logpost.argmax()

        MAPs[MAP_logpost] = {n.name : getattr(MAP_chain.getParams(), n.name)[MAP_idx] for i, n in enumerate(MAP_chain.getParamNames().names)}
        MAPs[MAP_logpost]["loglike"] = MAP_chain.loglikes[MAP_idx]

        files[MAP_logpost] = file

        if MAP_logpost > MAP_max_logpost:
            MAP = MAPs[MAP_logpost]
            MAP_chi2 = MAP_chain.loglikes[MAP_idx]
            MAP_max_logpost = MAP_logpost

    files = {k : files[k] for k in sorted(files, reverse=True)}
    if len(files) == 0:
        raise ValueError(f"Could not load any MAP chains using path template {MAP_path}.")
    print(f"Best MAP ({MAP_max_logpost:.4f}) in {list(files.values())[0]}")
    MAPs = {k : MAPs[k] for k in sorted(MAPs, reverse=True)}
    MAPs_array = np.array([[p for p in m.values()] for m in MAPs.values()])#[:len(MAPs)//3]
    MAPs_logpost = np.array([k for k in MAPs.keys()])#[:len(MAPs)//3]
    
    #print(MAPs)
    #print(MAPs_logpost, MAP_max_logpost, np.exp(MAPs_logpost-MAP_max_logpost))
    #print(MAPs_array)
    MAPs_wmean = np.average(MAPs_array, weights=np.exp(MAPs_logpost-MAP_max_logpost), axis=0)
    
    MAPs_wstd = np.sqrt(np.diag(np.cov(MAPs_array.T, aweights=np.exp(MAPs_logpost-MAP_max_logpost))))

    MAPs_range = np.vstack((MAPs_array.min(axis=0), MAPs_array.max(axis=0))).T
    MAPs_std = MAPs_array.std(axis=0)
    
    if verbose:
        for i, (k, v) in enumerate(MAP.items()):
            print(f"{k:<16}: {v:.3f}, range: ({MAPs_range[i][0]:.3f}, {MAPs_range[i][1]:.3f}), std: {MAPs_std[i]:.3f}, rel. err: {MAPs_std[i]/v:.3f}")
    
    MAP_std = {p : s for p, s in zip(MAP.keys(), MAPs_std)}
    MAP_wstd = {p : s for p, s in zip(MAP.keys(), MAPs_wstd)}
    MAP_wmean = {p : s for p, s in zip(MAP.keys(), MAPs_wmean)}
    
    for i, p in enumerate(MAP.keys()):
        try:
            stat_tools.weighted_median(MAPs_array[:,i], weights=np.exp(MAPs_logpost-MAP_max_logpost))
        except ValueError:
            print(p)
            print(MAPs_array[:,i])
            print(np.exp(MAPs_logpost-MAP_max_logpost))
    MAP_wmedian = {p : stat_tools.weighted_median(MAPs_array[:,i], weights=np.exp(MAPs_logpost-MAP_max_logpost)) for i, p in enumerate(MAP.keys())}
            
    return MAP, MAP_max_logpost, MAP_chi2, MAP_std, MAP_wmean, MAP_wstd, MAP_wmedian, MAPs

def get_stats(s, weights=None, params=None, CI_coverage=0.68, MAP=None):
    # Compute chain stats
    stats = {"PJ-HPDI"            : {},
             "chain MAP"          : {},
             "PJ-HPDI n_sample"   : {},
             "M-HPDI"             : {},
             "marg MAP"           : {},
             "M-HPDI constrained" : {},
             "std"                : {},
             "mean"               : {},
             "quantile CI"        : {},
             "median"             : {}, }

    chain_data = s.getParams()
    max_lnpost_idx = (chain_data.logpost).argmax()
    ln_post_sort_idx = np.argsort(chain_data.logpost)[::-1]

    params = params or [n.name for n in s.getParamNames().names]
    
    if weights is None:
        weights = s.weights
    
    for p in ["logpost", "logprior", "loglike", "weight"]:
        if p in params:
            params.pop(params.index(p))
            
    for p in params:        
        try:
            samples = getattr(chain_data, p)
        except AttributeError:
            continue
            
        if np.any(~np.isfinite(samples)):
            warnings.warn(f"NaNs in {p}. Skipping.")
            continue
        if np.isclose(np.var(samples), 0) and not np.isclose(np.mean(samples)**2, 0):
            warnings.warn(f"Variance of {p} close to zero. Skipping.")
            continue

        try:
            PJ_HPDI, chain_MAP, PJ_HPDI_n_sample = stat_tools.find_CI(method="PJ-HPD",
                                                                      samples=samples, weights=weights,
                                                                      coverage=CI_coverage, logpost_sort_idx=ln_post_sort_idx,
                                                                      return_point_estimate=True,
                                                                      return_extras=True,
                                                                      options={"strict" : True, "MAP" : MAP[p] if MAP is not None else None
                                                                              },
                                                                     )
        except RuntimeError as e:
            warnings.warn(f"Failed to get PJ-HPDI for parameter {p}. Trying one-sided interpolation.")
            PJ_HPDI = None
            chain_MAP = samples[max_lnpost_idx]
            PJ_HPDI_n_sample = 0
            
        if PJ_HPDI is None:
            try:
                PJ_HPDI, chain_MAP, PJ_HPDI_n_sample = stat_tools.find_CI(method="PJ-HPD",
                                                                          samples=samples, weights=weights,
                                                                          coverage=CI_coverage, logpost_sort_idx=ln_post_sort_idx,
                                                                          return_point_estimate=True,
                                                                          return_extras=True,
                                                                          options={"strict" : True, 
                                                                                   "MAP" : MAP[p] if MAP is not None else None,
                                                                                   "twosided" : False},
                                                                     )
            except RuntimeError as e:
                warnings.warn(f"Failed again to get PJ-HPDI.")

        if PJ_HPDI_n_sample < 30 and p in ["s8", "sigma8", "omegam", "h", "w", "omegak", "mnu"]:
            print(f"Number of PJ-HPD samples for parameter {p} less than 30: {PJ_HPDI_n_sample}")

        try:
            M_HPDI, marg_MAP, no_constraints = stat_tools.find_CI(method="HPD",
                                                                  samples=samples, weights=weights,
                                                                  coverage=CI_coverage,
                                                                  return_point_estimate=True,
                                                                  return_extras=True,
                                                                  options={"prior_edge_threshold" : 0.13}
                                                                  )
        except RuntimeError as e:
            warnings.warn(f"Failed to get M-HPDI for parameter {p}")

        stats["PJ-HPDI"][p] = PJ_HPDI
        stats["chain MAP"][p] = chain_MAP
        stats["PJ-HPDI n_sample"][p] = PJ_HPDI_n_sample

        stats["M-HPDI"][p] = M_HPDI
        stats["marg MAP"][p] = marg_MAP
        stats["M-HPDI constrained"][p] = not no_constraints

        stats["std"][p], stats["mean"][p] = stat_tools.find_CI(method="std",
                                                               samples=samples, weights=weights,
                                                               coverage=CI_coverage,
                                                               return_point_estimate=True,
                                                              )
        stats["quantile CI"][p], stats["median"][p] = stat_tools.find_CI(method="tail CI",
                                                                         samples=samples, weights=weights,
                                                                         coverage=CI_coverage,
                                                                         return_point_estimate=True,
                                                                         )

    stats["chain MAP_logpost"] = chain_data.logpost.max()
    if s.loglikes is not None:
        stats["chain MAP_loglike"] = s.loglikes[max_lnpost_idx]
    
    return stats

def is_systematics_chain(chain):
    return chain.chain_def["type"] == "systematics"

def is_blind(chain, blind):
    return chain.chain_def["blind"] == blind

def is_probe(chain, probe):
    return chain.chain_def["probes"] == probe

def get_chain_color(chain):
    return chain.chain_def["color"]

def get_chain_label(chain):
    return chain.chain_def["label"]

def select_chains(chains, selection, global_selection=None):
    selected_chains = [None]*len(selection)
    for c in chains.values():
        for i, selection_criteria in enumerate(selection):
            matches_all_selection_criteria = all([c.chain_def[k] == v for k,v in selection_criteria.items()])
            if matches_all_selection_criteria:
                selected_chains[i] = c
    
    [selected_chains.pop(i) for i, s in enumerate(selected_chains) if s is None]
        
    return selected_chains

def calculate_1d_tension(chain_pairs, file=sys.stdout):
    for chain_A, chain_B, options in chain_pairs:
        
        print("", file=file)
        print(f"{chain_A.name_tag} vs {chain_B.name_tag}", file=file)
        
        stats = options.get("stats", ["T", "H", "p_S"])
        params = options.get("params", ["s8", "sigma8", "Sigmaalpha"])
        
        tension_stats = {}
        
        if "T" in stats:
            tension_stats["T"] = {}
            
            fid = chain_A
            planck = chain_B
            for p in params:
                fid_mean = fid.chain_stats["mean"][p]
                fid_std = (fid.chain_stats["std"][p][1]-fid.chain_stats["std"][p][0])/2

                fid_marg_MAP = fid.chain_stats["marg MAP"][p]
                fid_marg_u_l = (fid.chain_stats["M-HPDI"][p][1]-fid.chain_stats["marg MAP"][p], 
                                fid.chain_stats["M-HPDI"][p][0]-fid.chain_stats["marg MAP"][p])

                planck_mean = planck.chain_stats["mean"][p]
                planck_std = (planck.chain_stats["std"][p][1]-planck.chain_stats["std"][p][0])/2

                planck_marg_MAP = planck.chain_stats["marg MAP"][p]
                planck_marg_u_l = (planck.chain_stats["M-HPDI"][p][1]-planck.chain_stats["marg MAP"][p], 
                                planck.chain_stats["M-HPDI"][p][0]-planck.chain_stats["marg MAP"][p])

                d = (fid_mean-planck_mean)/np.sqrt(fid_std**2 + planck_std**2)
                PTE = scipy.stats.norm.cdf(d)
                print(f"{p:<8} {fid_mean:.3f}±{fid_std:.3f} ({fid_marg_MAP:.3f}^+{fid_marg_u_l[0]:.3f}_{fid_marg_u_l[1]:.3f})"
                    f"  vs {planck_mean:.3f}±{planck_std:.3f} ({planck_marg_MAP:.3f}^+{planck_marg_u_l[0]:.3f}_{planck_marg_u_l[1]:.3f}): {PTE:.4f} {d:.3f}\\sigma", file=file)
        
                tension_stats["T"][p] = d
            
        if "H" in stats:
            tension_stats["H"] = {}
            
            print("", file=file)
            print("Hellinger distance S8", file=file)
            hellinger, hellinger_sigma = hellinger_distance_1D.hellinger_tension(
                                                                    sample1=chain_A.getParams().s8,
                                                                    sample2=chain_B.getParams().s8,
                                                                    weight1=chain_A.weights,
                                                                    weight2=chain_B.weights)
            print(f"d_H = {hellinger:.3f}, {hellinger_sigma:.2f}σ", file=file)
            tension_stats["H"]["s8"] = hellinger_sigma

            print("", file=file)
            print("Hellinger distance Sigma8", file=file)
            hellinger, hellinger_sigma = hellinger_distance_1D.hellinger_tension(
                                                                    sample1=chain_A.getParams().Sigmaalpha,
                                                                    sample2=chain_B.getParams().Sigmaalpha,
                                                                    weight1=chain_A.weights,
                                                                    weight2=chain_B.weights)
            print(f"d_H = {hellinger:.3f}, {hellinger_sigma:.2f}σ", file=file)
            tension_stats["H"]["Sigmaalpha"] = hellinger_sigma
            
        if "p_S" in stats:
            tension_stats["p_S"] = {}
            
            print("", file=file)
            # p_S statistic
            try:
                ew_A = process_chains.load_equal_weight_chain(chain_A)
                ew_B = process_chains.load_equal_weight_chain(chain_B)
            except KeyError:
                print(f"Could not find equal weight chains for pair {chain_A.name_tag} vs {chain_B.name_tag}")
                print("Trying getdist equal weighting for chain B")
                ew_B_idx = chain_B.thin_indices(options.get("thin_factor", 1))
                ew_B  = getdist.MCSamples(name_tag=chain_B.name_tag,
                                samples=chain_B.samples[ew_B_idx,:],
                                loglikes=chain_B.loglikes[ew_B_idx],
                                names=[n.name for n in chain_B.getParamNames().names],
                                sampler="nested",
                                ranges=chain_B.ranges)
                

            ew_A.sampler = "uncorrelated"
            ew_B.sampler = "uncorrelated"

    #         ew_A.addDerived(ew_A.getParams().sigma8*(ew_A.getParams().omegam/0.3)**0.58,
    #                      name="Sigmaalpha", label="\\Sigma_{8}")
    #         ew_B.addDerived(ew_B.getParams().sigma8*(ew_B.getParams().omegam/0.3)**0.58,
    #                      name="Sigmaalpha", label="\\Sigma_{8}")

            print("Samples in equal weight chain_A: ", ew_A.samples.shape[0])
            print("Samples in equal weight chain_B: ", ew_B.samples.shape[0])

            ew_diff_chain = tensiometer.mcmc_tension.parameter_diff_chain(ew_A, ew_B, boost=options.get("diff_chain_boost", 20))
            print("Samples in diff chain: ", ew_diff_chain.samples.shape[0])

            diff_params = options.get("diff_params", ["delta_s8"])
            prob = tensiometer.mcmc_tension.exact_parameter_shift(ew_diff_chain, 
                                                          param_names=diff_params,
                                                          method="nearest_elimination"
                                                          )
            prob_sigma = [np.sqrt(2)*scipy.special.erfcinv(1-_s) for _s in prob]

            print("", file=file)
            print(f"p_S statistic ({' '.join(diff_params)})", file=file)
            print(f"PTE: {1-prob[0]:.4f} ({1-prob[2]:.4f}-{1-prob[1]:.4f}), {prob_sigma[0]:.2f}σ ({prob_sigma[1]:.2f}σ-{prob_sigma[2]:.2f}σ)", file=file)
            tension_stats["p_S"]["s8"] = prob_sigma
            
        category = options.get("category", "tensions_1d")
        chain_A.chain_stats[category] = tension_stats

def calculate_advanced_chain_stats(c, file=sys.stdout):
    name = c.name_tag
    n_varied = c.chain_def["n_varied"]

    logL = c.loglikes

    d = 2 * (np.average(logL**2, weights=c.weights) - np.average(logL, weights=c.weights)**2)
    mean_logL = np.average(logL, weights=c.weights)
    log_meanL = scipy.special.logsumexp(logL, b=c.weights/np.sum(c.weights))

    # Raveri & Hu 2018
    u, l = list(c.ranges.upper.values())[:n_varied], list(c.ranges.lower.values())[:n_varied]

    # Flat priors
    prior_cov = np.diag([1/12*(u_-l_)**2 for u_,l_ in zip(u,l)])

    # Add covariances for Gaussian priors
    param_names = [n.name for n in c.getParamNames().names]
    #print(param_names)
    if "p_z1" in param_names:
        dz_idx = slice(param_names.index("p_z1"), param_names.index("p_z5")+1)
        dz_cov = np.loadtxt(os.path.join(c.chain_def["root_dir"], "data/correlate_dz/SOM_cov_multiplied.asc"))
        prior_cov[dz_idx,dz_idx] = dz_cov
    if "alpha_CIB" in param_names:
        prior_cov[:5,:5] *= 10 # Account for the artifically shrunk prior volume on the cosmology parameters
        calPlanck_idx = param_names.index("alpha_CIB")
        prior_cov[calPlanck_idx, calPlanck_idx] = (1.32e-6)**2

    posterior_cov = c.getCov(n_varied)
    # From tensionmetrics
    e, _ = np.linalg.eig(np.linalg.inv(prior_cov) @ posterior_cov)
    e[e > 1.] = 1.
    e[e < 0.] = 0.
    n_tot = len(e)
    n_eff = n_tot - np.sum(e)

    MAP_loglike = c.chain_stats["MAP_loglike"] if "MAP" in c.chain_stats else c.chain_stats["chain MAP_loglike"]
    
    MAP_loglike_var = np.var([v["loglike"] for v in c.chain_stats["MAP_runs"].values()]) if "MAP" in c.chain_stats else 0
    
    p_DIC1 = 2*MAP_loglike - 2*mean_logL
    p_DIC2 = d
    DIC1 = -2*MAP_loglike + 2*p_DIC1
    DIC2 = -2*MAP_loglike + 2*p_DIC2
    
    p_WAIC1 = 2*log_meanL - 2*mean_logL
    p_WAIC2 = d/2
    WAIC1 = -2*log_meanL + 2*p_WAIC1
    WAIC2 = -2*log_meanL + 2*p_WAIC2
    
    logZ = c.log_Z
    
    stats = {"d" : d, "n_eff" : n_eff, "n_post" : p_DIC1, "p_WAIC1" : p_WAIC1,
             "meanL" : mean_logL, "logZ" : logZ, "D" : mean_logL - logZ,
             "MAP_loglike" : MAP_loglike, "MAP_loglike_var" : MAP_loglike_var,
             "DIC1" : DIC1, "DIC2" : DIC2,
             "WAIC1" : WAIC1, "WAIC2" : WAIC2}

    print(f"{name:<20}: n_varied = {n_varied}, d = {d:.2f}, n_eff = {n_eff:.2f}, n_post = {p_DIC1:.2f}, n_WAIC = {p_WAIC1:.2f}, \n<logL> = {mean_logL:.2f}, D = {mean_logL - c.log_Z:.2f}, logZ = {c.log_Z:.2f}, \nDIC1 = {DIC1:.2f}, DIC2 = {DIC2:.2f}, WAIC1 = {WAIC1:.2f}, WAIC2 = {WAIC2:.2f}", file=file)


    chain_file = glob.glob(os.path.join(c.chain_def["root_dir"], "output/samples_*.txt"))[0]
    root = glob.glob(os.path.join(c.chain_def["root_dir"], "output/multinest/*_.txt"))[0][:-4]

    nested = anesthetic.NestedSamples(root=root)
    ns_output = nested.ns_output(nsamples=100)
    stats["nested"] = ns_output
    
    logL = nested.logL
    logdX = nested.dlogX(nsamples=100)
    logZ = nested.logZ(logdX)
        
    d = nested.d(logdX)
    logw = logdX.add(logL, axis=0) - logZ
    w = np.exp(logw)/np.sum(np.exp(logw))
    
    #print(mean_logL, log_meanL)
    
    mean_logL = np.sum(w*logL[:,None], axis=0)
    log_meanL = scipy.special.logsumexp(logL[:,None], b=w, axis=0)
    
    #print(mean_logL.mean(), mean_logL.std(), log_meanL.mean(), log_meanL.std())
    
    p_DIC1 = 2*MAP_loglike - 2*mean_logL
    p_DIC2 = d
    DIC1 = -2*MAP_loglike + 2*p_DIC1
    DIC2 = -2*MAP_loglike + 2*p_DIC2
    
    p_WAIC1 = 2*log_meanL - 2*mean_logL
    p_WAIC2 = d/2
    WAIC1 = -2*log_meanL + 2*p_WAIC1
    WAIC2 = -2*log_meanL + 2*p_WAIC2
    
    stats["nested"]["DIC1"] = DIC1
    stats["nested"]["DIC2"] = DIC2
    stats["nested"]["WAIC1"] = WAIC1
    stats["nested"]["WAIC2"] = WAIC2
    
    

#     print(f'log Z:  {ns_output["logZ"].mean():.2f}±{ns_output["logZ"].std():.2f}', file=file)
#     print(f'D:      {ns_output["D"].mean():.2f}±{ns_output["D"].std():.2f}', file=file)
#     print(f'd:      {ns_output["d"].mean():.2f}±{ns_output["d"].std():.2f}', file=file)

    return stats


def calculate_nd_tension(chain_triplets, file=sys.stdout):
    for chain_A, chain_B, chain_AB, options in chain_triplets:
        print(f"ND tension of {chain_A.name_tag}, {chain_B.name_tag}, {chain_AB.name_tag}")
        A = chain_A.chain_stats["advanced"]
        B = chain_B.chain_stats["advanced"]
        AB = chain_AB.chain_stats["advanced"]
        
        stats = options.get("stats", ["S", "Q_DMAP", "Q_UDM"])
        
        if "S" in stats:
            # Compute tension metrics
            logR = AB["logZ"] - A["logZ"] - B["logZ"]
            R = np.exp(logR)
            logI = A["D"] + B["D"] - AB["D"]
            logS = logR - logI

            d = A["d"] + B["d"] - AB["d"]
            PTE = scipy.stats.chi2(df=d).sf(d-2*logS)
            PTE_sigma = np.sqrt(2)*scipy.special.erfcinv(PTE)

            print("", file=file)
            print("Stats (chains)", file=file)
            print(f"log R: {logR:.2f}, R: {R:.2f}", file=file)
            print(f"log I: {logI:.2f}, log S: {logS:.2f}", file=file)
            print(f"d: {d:.2f}", file=file)
            print(f"PTE: {PTE:.3f}, {PTE_sigma:.2f}σ", file=file)

            logR = AB["nested"]["logZ"] - A["nested"]["logZ"] - B["nested"]["logZ"]
            R = np.exp(logR)
            w_joint = 1/(1+1/R)
            w_separate = 1/(1+R)
            logI = A["nested"]["D"] + B["nested"]["D"] - AB["nested"]["D"]
            logS = logR - logI

            d = A["nested"]["d"] + B["nested"]["d"] - AB["nested"]["d"]
            PTE = scipy.stats.chi2(df=d).sf(d-2*logS)
            PTE_sigma = np.sqrt(2)*scipy.special.erfcinv(PTE)

            print("", file=file)
            print("Stats (anesthetic)", file=file)
            print(f"log R: {logR.mean():.2f}±{logR.std():.2f}, R: {R.mean():.2f}±{R.std():.2f}", file=file)
            print(f"w_joint: {w_joint.mean():.2f}±{w_joint.std():.2f}, w_sep: {w_separate.mean():.2f}±{w_separate.std():.2f}", file=file)
            print(f"log I: {logI.mean():.2f}±{logI.std():.2f}, log S: {logS.mean():.2f}±{logS.std():.2f}", file=file)
            print(f"d: {d.mean():.2f}±{d.std():.2f}", file=file)
            print(f"PTE: {PTE.mean():.2f}±{PTE.std():.2f}, sigmas: {PTE_sigma.mean():.2f}±{PTE_sigma.std():.2f}", file=file)

        if "Q_DMAP" in stats:
            Q_DMAP = 2*A["MAP_loglike"] + 2*B["MAP_loglike"] - 2*AB["MAP_loglike"]
            n_eff = A["n_eff"] + B["n_eff"] - AB["n_eff"]
            Q_DMAP_PTE = scipy.stats.chi2(df=n_eff).sf(Q_DMAP)
            Q_DMAP_PTE_sigma = np.sqrt(2)*scipy.special.erfcinv(Q_DMAP_PTE)
            
            print("", file=file)
            print(f"Q_DMAP: {Q_DMAP:.2f}, n_eff: {n_eff:.2f}", file=file)
            print(f"PTE: {Q_DMAP_PTE:.3f}, {Q_DMAP_PTE_sigma:.2f}σ", file=file)
            
        if "Q_UDM" in stats:

            Q_UDM_params = options.get("Q_UDM_params", ["omegach2", "omegabh2", "h", "ns", "s8proxy"])
            Q_UDM_B_cutoff, _, _, _ = tensiometer.gaussian_tension.Q_UDM_get_cutoff(
                                                        chain_1=chain_B,
                                                        chain_2=chain_A,
                                                        chain_12=chain_AB,
                                                        param_names=Q_UDM_params)

            Q_UDM_B, Q_UDM_B_dof = tensiometer.gaussian_tension.Q_UDM(
                                                    chain_1=chain_B,
                                                    chain_12=chain_AB,
                                                    lower_cutoff=Q_UDM_B_cutoff,
                                                    param_names=Q_UDM_params)

            Q_UDM_B_PTE = scipy.stats.chi2(df=Q_UDM_B_dof).sf(Q_UDM_B)
            Q_UDM_B_PTE_sigma = np.sqrt(2)*scipy.special.erfcinv(Q_UDM_B_PTE)


            print("", file=file)
            print(f"Q_UDM (Planck vs joint): {Q_UDM_B:.2f}, n_eff: {Q_UDM_B_dof:.2f}", file=file)
            print(f"PTE: {Q_UDM_B_PTE:.3f}, {Q_UDM_B_PTE_sigma:.2f}σ", file=file)

def calculate_model_selection(models):
    w = {}
    for stat in ["DIC1", "WAIC1", "logZ"]:
        w[stat] = []
        delta = []
        for c in models:
            delta.append(np.array(c.chain_stats["advanced"]["nested"][stat]))

        for d_A in delta:
            f = -0.5 if stat in ["DIC1", "WAIC1"] else 1.0
            w[stat].append(1/np.sum([np.exp(f*(d_B - d_A)) for d_B in delta], axis=0))
               
    return w


def plot_marginal_density(chain, param, ax, plot_kwargs):
    density = chain.get1DDensity(param)
    density.normalize("integral")
    ax.plot(density.x, density.P, **plot_kwargs)


def get_violin_stats(chain, param):
    density = chain.get1DDensity(param)
    density.normalize("integral")
    mean = chain.chain_stats["mean"][param]
    median = chain.chain_stats["median"][param]
    quantiles = chain.chain_stats["M-HPDI"][param]
    violin_stats = {"coords": density.x,
                    "vals": density.P,
                    "mean": mean, "median": median,
                    # "quantiles": quantiles,
                    "min": quantiles[0], "max": quantiles[1]}
    return violin_stats



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--chain-dump")
    parse.add_argument("--ext-chain-dump")
    parse.add_argument("--make-plots", action="store_true")
    parse.add_argument("--make-stats", action="store_true")

    args = parse.parse_args()

    text_width = 523.5307/72
    column_width = 256.0748/72

    plot_settings = getdist.plots.GetDistPlotSettings()
    plot_settings.figure_legend_frame = False
    plot_settings.legend_frame = False
    plot_settings.figure_legend_loc = "upper right"
    plot_settings.alpha_filled_add=0.8
    plot_settings.alpha_factor_contour_lines=0.8
    plot_settings.fontsize = 8
    plot_settings.axes_fontsize = 8
    plot_settings.lab_fontsize = 8
    plot_settings.legend_fontsize = 8

    title_fontsize = 8
    legend_fontsize = 8
    label_fontsize = 8

    plot_settings.x_label_rotation = 45.0

    matplotlib.rc("text", usetex=True)
    matplotlib.rc("text.latex", preamble=r"""
    \usepackage{txfonts}
    %\newcommand{\mathdefault}[1][]{}
    """)

    matplotlib.rc("font", family="Times")

    make_plots = [
                    # "nofz_window_functions",
                    # "cov_contributions",
                    # "shearxCIB",
                    # "shearx100GHz",
                    # "shearxtSZ",
                    # "fid_constraints_Om_sigma8",
                    # "other_y_constraints_Om_sigma8",
                    # "other_y_constraints_Sigmaalpha",
                    # "logt_heat_constraint",
                    # "B-modes",
                    # "bp_vs_pCl",
                    # "bp_vs_pCl_window_functions",
                    # "bp_vs_pCl_constraints_Om_S8",
                    # "bp_vs_pCl_constraints_S8",
                    # "fid_constraints_all",
                    "ACT_Cls"
    ]


    # Chains
    base_dir_check_runs = "../runs/check_runs/"
    base_dir_final_runs_fast = "../runs/final_runs_fast/"
    base_dir_final_runs_long = "../runs/final_runs_long/"
    base_dir_final_runs = "../runs/final_runs/"
    base_dir_MAP_runs = "../runs/MAP_runs/"

    kcap_path = "../../../KiDS/kcap/"
            
    chain_def_cosmic_shear =  [
            {"root_dir" : os.path.join(kcap_path,
                                       "runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE"),
             "name"     : "EE_bp",
             "label"    : "Asgari et al. (2021) bandpowers",
             "n_varied" : 12,
             "n_data"   : 120,
             "color"    : "C1"},
            {"root_dir" : os.path.join(base_dir_check_runs, 
                                      "cosmic_shear_run3_hmcode_old_m_multinest/"),
              "name"     : "EE_hmcode_old_m",
              "label"    : "pseudo-$C_\\ell$, {\\sc HMCode-2016}",
              "n_varied" : 12,
              "n_data"   : 120,
              "color"    : "C0"},
            {"root_dir" : os.path.join(base_dir_check_runs, 
                                      "cosmic_shear_run5_hmx_old_m_multinest/"),
              "name"     : "EE_hmx_old_m",
              "label"    : "pseudo-$C_\\ell$, {\\sc HMx}",
              "n_varied" : 12,
              "n_data"   : 120,
              "color"    : "C0"},
            {"root_dir" : os.path.join(base_dir_check_runs, 
                                      "cosmic_shear_run6_hmcode2020_old_m_multinest/"),
              "name"     : "EE_hmcode2020_old_m",
              "label"    : "pseudo-$C_\\ell$, {\\sc HMCode-2020}",
              "n_varied" : 12,
              "n_data"   : 120,
              "color"    : "C0"},
    ]

    chain_def_fid = [
              {"root_dir" : os.path.join(base_dir_final_runs, 
                                         "TE_fid/"),
              "MAP_path" : [os.path.join(base_dir_MAP_runs, "MAP_TE_fid/MAP_start_idx_*/output/samples_MAP_start_idx_*.txt")],
              "name"     : "TE_fid",
              "label"    : "shear--tSZ",
              "n_varied" : 13,
              "n_data"   : 40,
              "color"    : "C0"},
             {"root_dir" : os.path.join(base_dir_final_runs, 
                                         "EE_fid/"),
              "MAP_path" : [os.path.join(base_dir_MAP_runs, "MAP_EE_fid/MAP_start_idx_*/output/samples_MAP_start_idx_*.txt")],
              "name"     : "EE_fid",
              "label"    : "Cosmic shear",
              "n_varied" : 12,
              "n_data"   : 120,
              "color"    : "C0"},
             {"root_dir" : os.path.join(base_dir_final_runs_long, 
                                        "joint_fid/"),
              "MAP_path" : [os.path.join(base_dir_MAP_runs, "MAP_joint_fid/MAP_start_idx_*/output/samples_MAP_start_idx_*.txt")],
              "name"     : "joint_fid",
              "label"    : "Cosmic shear + shear--tSZ",
              "n_varied" : 13,
              "n_data"   : 160,
              "color"    : "C0"},
    ]

    chain_def_other_y_maps = [
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "y_milca_nocib_marg/"),
         "name"     : "y_milca_nocib_marg",
         "label"    : "milca, no CIB marg",
         "n_varied" : 12,
         "n_data"   : 40,
         "color"    : "C0"},
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "y_nilc_nocib_marg/"),
         "name"     : "y_nilc_nocib_marg",
         "label"    : "nilc, no CIB marg",
         "n_varied" : 12,
         "n_data"   : 40,
         "color"    : "C0"},
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "y_yan2019_nocib_nocib_marg/"),
         "name"     : "y_yan2019_nocib_nocib_marg",
         "label"    : "Yan 2019 CIB subtracted, no CIB marg",
         "n_varied" : 12,
         "n_data"   : 40,
         "color"    : "C0"},
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "y_yan2019_nocib_beta1.2_nocib_marg/"),
         "name"     : "y_yan2019_nocib_beta1.2_nocib_marg",
         "label"    : "Yan 2019 CIB subtracted, beta=1.2, no CIB marg",
         "n_varied" : 12,
         "n_data"   : 40,
         "color"    : "C0"},

        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "y_ACT_BN_nocib_marg/"),
         "name"     : "y_ACT_nocib_marg",
         "label"    : "ACT, no CIB marg",
         "n_varied" : 12,
         "n_data"   : 40,
         "color"    : "C0"},
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "y_ACT_BN_nocib_nocib_marg/"),
         "name"     : "y_ACT_nocib_nocib_marg",
         "label"    : "ACT, CIB deprojected, no CIB marg",
         "n_varied" : 12,
         "n_data"   : 40,
         "color"    : "C0"},
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "y_ACT_BN_nocmb_nocib_marg/"),
         "name"     : "y_ACT_nocmb_nocib_marg",
         "label"    : "ACT, CMB deprojected, no CIB marg",
         "n_varied" : 12,
         "n_data"   : 40,
         "color"    : "C0"},
    ]

    chain_def_misc = [
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "TE_no_y_IA/"),
         "name"     : "TE_no_y_IA",
         "label"    : "TE, no tSZ IA",
         "n_varied" : 13,
         "n_data"   : 40,
         "color"    : "C0"},
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "joint_no_y_IA/"),
         "name"     : "joint_no_y_IA",
         "label"    : "joint, no tSZ IA",
         "n_varied" : 13,
         "n_data"   : 160,
         "color"    : "C0"},
        
        {"root_dir" : os.path.join(base_dir_final_runs_fast, 
                                   "TE_no_SSC/"),
         "name"     : "TE_no_SSC",
         "label"    : "TE no SSC",
         "n_varied" : 13,
         "n_data"   : 40,
         "color"    : "C0"},
    ]

    chain_def_Planck = [
                # Planck
                {"root_dir" : os.path.join(kcap_path,
                                       "runs/3x2pt/Planck/multinest_Planck"),
                "name"     : "Planck_fiducial",
                "label"    : "Planck 2018 TTTEEE+lowE",
                "blind"    : "None",
                "probes"   : ("CMB",),
                "n_varied" : 7,
                "MAP_path" : [os.path.join(kcap_path,
                                       "runs/3x2pt/data_iterated_cov_MAP/cosmology_with_Planck/MAP_*_blindC_Planck_Powell/chain/samples_MAP_*_blindC_Planck_Powell.txt")],
                "type"     : "cosmology",
                "color"    : "darkslategrey"},
    ]



    chains = {}

    CI_coverage = 0.68

    defs = [
        *chain_def_cosmic_shear,
        *chain_def_fid,
        *chain_def_other_y_maps,
        *chain_def_misc,
        *chain_def_Planck,
        ]

    if args.chain_dump is not None and os.path.isfile(args.chain_dump):
        with open(args.chain_dump, "rb") as f:
            chains = pickle.load(f)
    else:
        for c in defs:
            print(f"Chain {c['name']}")

            # Load chain
            chain_file = glob.glob(os.path.join(c["root_dir"], "output/samples_*.txt")) + glob.glob(os.path.join(c["root_dir"], "chain/samples_*.txt"))
            if len(chain_file) != 1:
                raise ValueError(f"Could not find unique chain file in {c['root_dir']}")
            chain_file = chain_file[0]

            value_file = os.path.join(c["root_dir"], "config/values.ini")

            s = process_chains.load_chain(chain_file, values=value_file, run_name=c["name"], 
                                        strict_mapping=False, ignore_inf=True)

            param_names = [n.name for n in s.getParamNames().names]
                                
            if "omegamh2" not in param_names:
                s.addDerived(s.getParams().omegam*s.getParams().h**2,
                            name="omegamh2", label="\\Omega_{\\rm m} h^2")
            if "omeganuh2" not in param_names and "omeganu" in param_names:
                s.addDerived(s.getParams().omeganu*s.getParams().h**2,
                            name="omeganuh2", label="\\Omega_{\\nu} h^2")
            if "Sigmaalpha" not in [n.name for n in s.getParamNames().names]:
                s.addDerived(s.getParams().sigma8*(s.getParams().omegam/0.3)**0.58,
                            name="Sigmaalpha", label="\\Sigma_{8}")
            if "Sigma0p2" not in [n.name for n in s.getParamNames().names]:
                s.addDerived(s.getParams().sigma8*(s.getParams().omegam/0.3)**0.2,
                     name="Sigma0p2", label="\\Sigma_{8}^{\\alpha=0.2}")

            s.chain_def = c
            chains[c["name"]] = s

            stats = {}
            if "MAP_path" in c:
                stats["MAP"], stats["MAP_logpost"], stats["MAP_loglike"], \
                stats["MAP_std"], stats["MAP_wmean"], stats["MAP_wstd"], \
                stats["MAP_wmedian"], stats["MAP_runs"] = get_MAP(c["MAP_path"], verbose=True)

            stats = {**stats, **get_stats(s, CI_coverage=CI_coverage, MAP=stats["MAP"] if "MAP" in stats else None)}
            if "MAP" in stats:
                print("Improvement in logpost from chain to MAP: {:.2f} -> {:.2f}".format(stats["chain MAP_logpost"], stats["MAP_logpost"]))
                print("Fractional error from scatter between MAP runs")
                for p in ["loglike", "s8", "a_ia", "logt_heat"]:
                    if p not in stats["MAP"] or p not in stats["std"]:
                        continue
                    std = (stats["std"][p][1]-stats["std"][p][0])/2
                    print(f'  {p:<10}: {stats["MAP"][p]:.3f}, {stats["MAP_wmedian"][p]:.3f}, {stats["MAP_std"][p]:.3f}, {stats["MAP_std"][p]/std:.3f}')
                if stats["chain MAP_logpost"] > stats["MAP_logpost"]:
                    warnings.warn(f"MAP has not improved for chain {c['name']}")
            chains[c["name"]].chain_stats = stats

        if args.chain_dump is not None:
            with open(args.chain_dump, "wb") as f:
                pickle.dump(chains, f)

    if args.make_plots:
        n_ell_bin = 12
        n_z_bin = 5

        field_idx_EE = [(i, j) for i in range(n_z_bin)
                            for j in range(i+1)]
        field_idx_TE = [(i, 0) for i in range(n_z_bin)]
        ell_eff = np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/data/Cl_TE_shear_KiDS1000_gal_y_milca.txt", usecols=[0])

        if "nofz_window_functions" in make_plots:
            print("Plotting nofz_window_functions")
            import pyccl as ccl
            ccl_cosmo = ccl.CosmologyVanillaLCDM()

            prediction_path = "../runs/theory_prediction_runs/cov_theory_predictions_run3_hmx_nocib_beam1.6/output/data_block/"        # noqa: E501

            z = np.loadtxt(os.path.join(prediction_path, "nz_kids1000/z.txt"))
            lensing_nz = [(z, np.loadtxt(os.path.join(prediction_path, f"nz_kids1000/bin_{i+1}.txt")))      # noqa: E501
                        for i in range(5)]

            WL_tracers = []
            for (z, nz) in lensing_nz:
                WL_tracers.append(ccl.WeakLensingTracer(ccl_cosmo,
                                                        dndz=(z, nz)))
            tSZ_tracer = ccl.tSZTracer(ccl_cosmo)

            W_y = ccl.pyutils._get_spline1d_arrays(tSZ_tracer._trc[0].kernel.spline)
            W_gamma = [ccl.pyutils._get_spline1d_arrays(trc._trc[0].kernel.spline) for trc in WL_tracers]

            W_y = (1/ccl.scale_factor_of_chi(ccl_cosmo, W_y[0])-1, W_y[1])
            W_gamma = [(1/ccl.scale_factor_of_chi(ccl_cosmo, trc[0])-1, trc[1]) for trc in W_gamma]

            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(column_width, column_width))
            fig.subplots_adjust(left=0.2, bottom=0.13, hspace=0, right=0.97, top=0.98)

            z_bins = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.2)]
            [ax[0].plot(*nofz, label=f"${z[0]} < z_\\mathrm{{B}} \\leq {z[1]}$") for i, (nofz, z) in enumerate(zip(lensing_nz, z_bins))]
            [ax[0].axvspan(*z_bin, facecolor=f"C{i}", alpha=0.3) for i, z_bin in enumerate(z_bins)]
            ax[0].set_xlim(0, 2.0)
            ax[0].set_ylim(bottom=0)
            ax[0].set_ylabel(r"$n(z)$")
            ax[0].legend(frameon=False, fontsize=legend_fontsize)

            norm = W_gamma[-1][1].max()
            [ax[1].plot(chi, W/norm) for (chi, W) in W_gamma]

            z, W = W_y
            norm = W.max()
            ax[1].plot(z, W/norm, c="C6", ls="--", label="Compton-$y$")

            ax[1].set_ylim(bottom=0, top=1.2)
            ax[1].set_ylabel(r"$W(z)$ [arbitrary units]")

            ax[1].set_xlabel(r"$z$")
            ax[1].legend(frameon=False, fontsize=legend_fontsize)

            fig.savefig("plots/nofz_Wofz.pdf")


        if "cov_contributions" in make_plots:
            print("Plotting cov_contributions")
            import matplotlib.colors

            cov_joint = {"total" : np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/cov/covariance_total_SSC_mask_joint.txt"),
                        "gaussian" : np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/cov/covariance_total_gaussian_joint.txt"),
                        "SSC" : np.loadtxt("../data/xcorr/cov/NG_cov_Planck/covariance_hmx_SSC_mask_wl_joint.txt"),
                        "T" : np.loadtxt("../data/xcorr/cov/NG_cov_Planck/covariance_hmx_cNG_1h_joint.txt"),
                    }
            cov_joint["NG"] = cov_joint["total"] - cov_joint["gaussian"]

            m = np.tile([0,0,1,1,1,1,1,1,1,1,0,0], 20).astype(dtype=bool)
            cov_joint = {k: c[np.ix_(m, m)] for k, c in cov_joint.items()}

            s = {k: np.diag(np.sqrt(1/np.diag(c))) for k, c in cov_joint.items()}
            corr = {k: s[k] @ c @ s[k] for k, c in cov_joint.items()}

            fig, ax = plt.subplots(1, 1, figsize=(column_width, column_width))
            fig.subplots_adjust(left=0.1, bottom=0.08, top=0.9)

            mask = np.zeros_like(corr["total"], dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = True
            c1 = np.ma.masked_array(corr["total"], mask)

            # c2 = np.ma.masked_array(np.abs(cov_joint["NG"])/np.abs(cov_joint["gaussian"]), ~mask)
            c2 = np.ma.masked_array(np.abs(cov_joint["SSC"])/np.abs(cov_joint["total"]), ~mask)

            imshow_c1 = ax.imshow(c1, cmap=plt.get_cmap("viridis"), norm=matplotlib.colors.Normalize(vmax=1), interpolation="nearest")
            imshow_c2 = ax.imshow(c2, cmap=plt.get_cmap("plasma"),
                                # norm=matplotlib.colors.Normalize(vmin=0, vmax=2)
                                norm=matplotlib.colors.LogNorm(),
                                interpolation="nearest"
                                )

            cb1 = plt.colorbar(imshow_c1, location="bottom", pad=0.005, anchor=(0.0, 1.0), shrink=0.83)
            cb2 = plt.colorbar(imshow_c2, shrink=0.96, pad=0.02)
            cb1.ax.tick_params(labelsize=legend_fontsize) 
            cb2.ax.tick_params(labelsize=legend_fontsize) 

            ax.tick_params(which="major", bottom=False, left=True, top=True)
            ax.tick_params(which="minor", bottom=False, left=False, top=False)
            ax.tick_params(which="major", labelbottom=False, labelleft=False, labeltop=False)
            ax.tick_params(which="minor", labelbottom=False, labelleft=True, labeltop=True)

            ax.set_xticks([0, 120, 159])
            ax.set_xticks([60, 140], minor=True)
            ax.set_yticks([0, 120, 159])
            ax.set_yticks([60, 140], minor=True)
            ax.set_xticklabels([r"$C^{\gamma\gamma}_\ell$", r"$C^{\gamma y}_\ell$"], minor=True)
            ax.set_yticklabels([r"$C^{\gamma\gamma}_\ell$", r"$C^{\gamma y}_\ell$"], minor=True, rotation="vertical", va="center")


            cb1.set_label("Correlation coefficient", fontsize=legend_fontsize)
            cb2.set_label("SSC/total", fontsize=legend_fontsize)

            fig.dpi = 300
            fig.savefig("plots/covariance_contributions_joint.pdf")

        if "shearxCIB" in make_plots:
            print("Plotting shearxCIB")
            from CIB_model import CIBModel

            Cl_CIB = {"TE": np.loadtxt("../results/measurements/shear_KiDS1000_545GHz_CIB/likelihood/data/Cl_TE_shear_KiDS1000_gal_545GHz_CIB.txt")[:, 1:],
                    "TB": np.loadtxt("../results/measurements/shear_KiDS1000_545GHz_CIB/likelihood/data/Cl_TB_shear_KiDS1000_gal_545GHz_CIB.txt")[:, 1:]}
            Cl_CIB_cov = {"TE": np.loadtxt("../results/measurements/shear_KiDS1000_545GHz_CIB/likelihood/cov/covariance_gaussian_nka_TETE.txt"),
                        }
            Cl_CIB_err = {"TE": np.sqrt(np.diag(Cl_CIB_cov["TE"])).reshape(-1, n_ell_bin).T,
                        }

            scaling_factor = 1e5 * ell_eff**2/(2*np.pi)
            Y = scaling_factor[:, None] * Cl_CIB["TE"]
            S = np.diag(np.tile(scaling_factor, Y.shape[1]))

            Y_cov = S @ Cl_CIB_cov["TE"] @ S
            X = np.log10(ell_eff)

            CIB_model = CIBModel(X, Y, Y_cov)
            CIB_model.load_state(state_file="../results/measurements/shear_KiDS1000_545GHz_CIB/GP_model/GP_state.torchstate")
            # CIB_model.print_model_parameters()

            ell_CIB_pred = np.arange(51, 2953)
            CIB_prediction, CI = CIB_model.predict(np.log10(ell_CIB_pred), CI=True)
            # Undo normalisation to get Cl
            CIB_prediction *= 1/(1e5 * ell_CIB_pred**2/(2*np.pi))[:, None]
            CIB_prediction_CI_l = CI[0] * 1/(1e5 * ell_CIB_pred**2/(2*np.pi))[:, None]
            CIB_prediction_CI_u = CI[1] * 1/(1e5 * ell_CIB_pred**2/(2*np.pi))[:, None]

            fig, ax = plotting_utils.plot_xcorr(
                Cls=[{"name"        : "Gaussian process model CI",
                    "label"       : None,
                    "X"           : ell_CIB_pred,
                    "Y"           : CIB_prediction,
                    "Y_lower"     : CIB_prediction_CI_l,
                    "Y_upper"     : CIB_prediction_CI_u,
                    "plot_kwargs" : {"facecolor": "C1", "alpha": 0.5}},
                    {"name"        : "Gaussian process\nmodel",
                    "X"           : ell_CIB_pred,
                    "Y"           : CIB_prediction,
                    "plot_kwargs" : {"c": "C1"}},
                    {"name"        : "pseudo-$C_\\ell$",
                    "X"           : ell_eff,
                    "Y"           : Cl_CIB["TE"],
                    "Y_error"     : Cl_CIB_err["TE"],
                    "plot_kwargs" : {"c": "C0", "ls": "none", "marker": "."}},
                    
                    ],
                n_z_bin=5,
                figsize=(column_width, 0.7*column_width),
                scaling=lambda x: x**2/(2*np.pi), x_offset=None,
                x_data_range=(100, 1500), x_range=(100, 1800), sharey=True,
                y_range=(-1.5e-5, 2.9e-5), field="CIB",
                y_label=r"$\ell^2/2\pi\ C_\ell^{\gamma I_\mathrm{CIB}}$",
                legend_fontsize=legend_fontsize, label_fontsize=legend_fontsize,
            )
            ax[0,2].get_xticklabels()[1].set_visible(False)
            fig.savefig("plots/data_vectors_CIB.pdf")

        if "shearx100GHz" in make_plots:
            print("Plotting shearx100GHz")
            Cl_100GHz = {"TE": np.loadtxt("../results/measurements/shear_KiDS1000_100GHz_HFI/likelihood/data/Cl_TE_shear_KiDS1000_gal_100GHz_HFI.txt")[:, 1:]}
            Cl_100GHz_err = {"TE": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_100GHz_HFI/likelihood/cov/covariance_gaussian_nka_TETE.txt"))).reshape(-1, n_ell_bin).T}

            fig, ax = plotting_utils.plot_xcorr(
                Cls=[{"name"        : "pseudo-$C_\\ell$",
                    "X"           : ell_eff,
                    "Y"           : Cl_100GHz["TE"],
                    "Y_error"     : Cl_100GHz_err["TE"],
                    "plot_kwargs" : {"c": "C0", "ls": "none", "marker": "."}},
                    
                    ],
                n_z_bin=5,
                figsize=(column_width, 0.7*column_width),
                scaling=lambda x: x**2/(2*np.pi),
                x_data_range=(100, 1500), x_range=(100, 1800), sharey=True,
                y_range=(-7e-8, 7e-8),
                field="100\,GHz",
                y_label=r"$\ell^2/2\pi\ C_\ell^{\gamma I_\mathrm{100\,GHz}}$",
                legend_fontsize=legend_fontsize, label_fontsize=legend_fontsize,
            )
            ax[0,2].get_xticklabels()[1].set_visible(False)
            fig.savefig("plots/data_vectors_100GHz.pdf")

        if "shearxtSZ" in make_plots:
            print("Plotting shearxtSZ")
            ppd_dir = "../runs/final_runs/joint_fid/output/ppd/"

            tpd_Cls = np.load(os.path.join(ppd_dir, "tpd_Cls.npz"))
            ppd_Cls = np.load(os.path.join(ppd_dir, "ppd_Cls.npz"))
            ppd_params = np.load(os.path.join(ppd_dir, "ppd_params.npz"))
            sort_idx = np.argsort(ppd_params["logt_heat"])

            ell_eff = np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/data/Cl_TE_shear_KiDS1000_gal_y_milca.txt", usecols=[0])

            Cl = {"EE": np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/data/Cl_EE_shear_KiDS1000_gal.txt")[:, 1:],
                "TE": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/data/Cl_TE_shear_KiDS1000_gal_y_milca.txt")[:, 1:],
                }

            Cl_err = {"EE": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/cov/covariance_total_SSC_mask_EEEE.txt"))).reshape(-1, n_ell_bin).T,
                    "TE": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/cov/covariance_total_SSC_mask_TETE.txt"))).reshape(-1, n_ell_bin).T,
                    }
                    
            cmap = plt.get_cmap("inferno")

            plotting_utils.plot_joint(
                Cls_EE=[{"name"        : "PPD",
                        "X"           : ell_eff,
                        "Y"           : ppd_Cls["EE"][sort_idx],
                        "PPD_percentiles": [(16, 84), (2.5, 97.5)],
                        #  "PPD_percentiles": [(16, 84)],
                        "plot_kwargs" : {"facecolor": "grey", "alpha": 0.5}},
                        {"name"        : "TPD",
                        "X"           : ell_eff,
                        "Y"           : tpd_Cls["EE"][sort_idx],
                        "PPD_color_param": ppd_params["logt_heat"][sort_idx],
                        "plot_kwargs" : {"cmap": cmap, "ls": "-", "alpha": 0.5, "lw": 0.5}},
                        {"name"        : "pseudo-$C_\\ell$",
                        "X"           : ell_eff,
                        "Y"           : Cl["EE"], 
                        "Y_error"     : Cl_err["EE"],
                        "plot_kwargs" : {"c": "k", "ls": "none", "marker": "."}},
                    ],
                Cls_TE=[{"name"        : "PPD",
                        "X"           : ell_eff,
                        "Y"           : ppd_Cls["TE"][sort_idx],
                        "PPD_percentiles": [(16, 84), (2.5, 97.5)],
                        #  "PPD_percentiles": [(16, 84)],
                        "plot_kwargs" : {"facecolor": "grey", "alpha": 0.5}},
                        {"name"        : "TPD",
                        "X"           : ell_eff,
                        "Y"           : tpd_Cls["TE"][sort_idx],
                        "PPD_color_param": ppd_params["logt_heat"][sort_idx],
                        "plot_kwargs" : {"cmap": cmap, "ls": "-", "alpha": 0.5, "lw": 0.5}},
                        {"name"        : "MILCA",
                        "X"           : ell_eff,
                        "Y"           : Cl["TE"],
                        "Y_error"     : Cl_err["TE"],
                        "plot_kwargs" : {"c": "k", "ls": "none", "marker": "."}},
                    ],
                n_z_bin=n_z_bin,
                figsize=(text_width, 0.9*text_width), x_offset=None,
                scaling_EE=lambda x: x, scaling_TE=lambda x: x**2/(2*np.pi),
                x_data_range=(100, 1500), x_range=(100, 1800), sharey=True,
                y_range_EE=(-4e-7, 9e-7), y_range_TE=(-5e-10, 1.9e-9),
                y_label_EE=r"$\ell C_\ell^{\gamma\gamma}$", y_label_TE=r"$\ell^2/2\pi\ C_\ell^{\gamma y}$",
                legend_fontsize=legend_fontsize, label_fontsize=legend_fontsize,
                filename="plots/data_vectors_w_model.pdf"
            )

        if "fid_constraints_Om_sigma8" in make_plots:
            print("Plotting fid_constraints_Om_sigma8")
            width = column_width
            g = getdist.plots.get_subplot_plotter(width_inch=width, scaling=False,
                                                settings=copy.deepcopy(plot_settings))

            g.make_figure(nx=1, ny=1, sharex=True, sharey=True)
            ax = []
            chain_selection = [{"name" : "EE_fid",},
                               {"name" : "TE_fid",},
                               {"name" : "joint_fid",},
                               {"name" : "Planck_fiducial",},
            ]
            chain_labels = ["Cosmic shear", "shear--tSZ", "Cosmic shear + shear--tSZ", "Planck"]

            chains_to_plot = select_chains(chains, chain_selection)

            print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")

            ax.append(g._subplot(0, 0))
            g.plot_2d(chains_to_plot, param1="omegam", param2="sigma8", 
                    filled=True,
                    add_legend_proxy=True,
                    lims=[0.1, 0.65,
                          0.4, 1.1],
                    colors=["C0", "C1", "C2", "C3", "C4"],
                    ax=ax[-1]
            )
            ax[-1].legend(g.contours_added[-len(chains_to_plot):],
                        chain_labels,
                        frameon=False, fontsize=legend_fontsize)
            g.export("plots/KiDSxPlanck_fid_CIB_Om_sigma8_w_Planck.pdf")

        if "other_y_constraints_Om_sigma8" in make_plots:
            print("Plotting other_y_constraints_Om_sigma8")
            width = column_width
            g = getdist.plots.get_subplot_plotter(width_inch=width, scaling=False,
                                                settings=copy.deepcopy(plot_settings))

            g.make_figure(nx=1, ny=2, ystretch=0.7, sharex=True, sharey=True)
            ax = []

            chain_selection = [
                                {"name" : "TE_fid",},
                                {"name" : "y_milca_nocib_marg",},
                                {"name" : "y_nilc_nocib_marg",},
                                {"name" : "y_yan2019_nocib_nocib_marg",},
                                {"name" : "y_yan2019_nocib_beta1.2_nocib_marg",},
            ]
            chains_to_plot = select_chains(chains, chain_selection)
            chain_labels = ["\\textit{Planck} MILCA, fiducial",
                            "\\textit{Planck} MILCA, no CIB marginalisation",
                            "\\textit{Planck} NILC",
                            "Yan et al. (2019), CIB subtracted",
                            "Yan et al. (2019), CIB subtracted, $\\beta=1.2$",
                            ]
            filled = [True, False, False, False, False]

            print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")
            ax.append(g._subplot(0, 0))
            g.plot_2d(chains_to_plot, param1="omegam", param2="sigma8", 
                    filled=filled,
                    add_legend_proxy=True,
                    colors=["C1", "C3", "C4", "C5", "C6"],
                    ax=ax[-1]
            )
            ax[-1].legend(g.contours_added[-len(chains_to_plot):],
                        chain_labels,
                        loc=2,
                        frameon=False, fontsize=legend_fontsize)

            chain_selection = [
                            {"name" : "TE_fid",},
                            {"name" : "y_ACT_nocib_marg",},
                            {"name" : "y_ACT_nocmb_nocib_marg",},
                            {"name" : "y_ACT_nocib_nocib_marg",},
            ]
            chains_to_plot = select_chains(chains, chain_selection)
            chain_labels = ["\\textit{Planck} MILCA, fiducial",
                            "ACT",
                            "ACT, CMB deprojected",
                            "ACT, CIB deprojected",
                            ]
            filled = [True, False, False, False]

            print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")
            ax.append(g._subplot(0, 1, sharex=ax[0], sharey=ax[0]))
            g.plot_2d(chains_to_plot, param1="omegam", param2="sigma8", 
                    filled=filled,
                    add_legend_proxy=True,
                    colors=["C1", "C7", "C8", "C9", "C10"],
                    lims=[0.1, 0.65,
                          0.4, 1.2],
                    ax=ax[-1]
            )
            ax[-1].legend(g.contours_added[-len(chains_to_plot)+1:],
                        chain_labels[1:],
                        loc=2,
                        frameon=False, fontsize=legend_fontsize)

            g.gridspec.update(hspace=0)
            g.export("plots/KiDSxothers_Om_sigma8.pdf")

        if "other_y_constraints_Sigmaalpha" in make_plots:
            print("Plotting other_y_constraints_Sigmaalpha")
            fig, ax = plt.subplots(1, 1, figsize=(column_width, 1.0*column_width))
            fig.subplots_adjust(top=0.98, left=0.02, right=0.98, bottom=0.2)

            chain_selection = [
                                {"name" : "TE_fid",},
                                {"name" : "y_milca_nocib_marg",},
                                {"name" : "y_nilc_nocib_marg",},
                                {"name" : "y_yan2019_nocib_nocib_marg",},
                                {"name" : "y_yan2019_nocib_beta1.2_nocib_marg",},
                                {"name" : "y_ACT_nocib_marg",},
                                {"name" : "y_ACT_nocmb_nocib_marg",},
                                {"name" : "y_ACT_nocib_nocib_marg",},
            ][::-1]
            chains_to_plot = select_chains(chains, chain_selection)
            chain_labels = ["\\textit{Planck} MILCA, fiducial",
                            "\\textit{Planck} MILCA, no CIB marginalisation",
                            "\\textit{Planck} NILC",
                            "Yan et al. (2019), CIB subtracted",
                            "Yan et al. (2019), CIB subtracted, $\\beta=1.2$",
                            "ACT",
                            "ACT, CMB deprojected",
                            "ACT, CIB deprojected",
                            ][::-1]
            chain_colors = ["C1",
                            "C3", "C4", "C5", "C6",
                            "C7", "C8", "C9"][::-1]

            fid_chain = "TE_fid"
            param = "Sigma0p2"

            fid_CI = chains[fid_chain].chain_stats["M-HPDI"][param]
            fid_MAP = chains[fid_chain].chain_stats["marg MAP"][param]

            ax.axvline(fid_CI[0], color="C1", lw=1, alpha=0.5)
            ax.axvline(fid_CI[1], color="C1", lw=1, alpha=0.5)
            ax.axvline(fid_MAP, color="C1", ls=":", lw=1)

            violin_stats = [get_violin_stats(c, param) for c in chains_to_plot]

            violin_plots = ax.violin(violin_stats, vert=False, showmeans=False, showextrema=False)

            for i, chain in enumerate(chains_to_plot):
                violin_plots["bodies"][i].set_facecolor(chain_colors[i])

                MAP = chain.chain_stats["marg MAP"][param]
                l, u = chain.chain_stats["M-HPDI"][param]
                err = np.array([MAP-l, u-MAP])[:, None]
                ax.errorbar(x=MAP, y=i+1, xerr=err, marker=".",  c=chain_colors[i], capsize=3, capthick=1)
                ax.text(0.42, i+1, chain_labels[i], fontsize=legend_fontsize)

            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_xlabel(r"$\Sigma_8^{\alpha=0.2} = \sigma_8 \left(\frac{\Omega_\mathrm{m}}{0.3}\right)^{0.2}$", x=0.8)
            ax.set_xlim(0.41, 0.83)
            ax.set_xticks([0.65, 0.7, 0.75, 0.8])
            ax.set_xticklabels([0.65, 0.7, 0.75, 0.8])

            fig.savefig("plots/other_y_maps_violin_Sigmaalpha.pdf")
            
        if "logt_heat_constraint" in make_plots:
            print("Plotting logt_heat_constraint")
            fig, ax = plt.subplots(1, 1, sharex=True, figsize=(column_width, 0.9*column_width))
            fig.subplots_adjust(hspace=0, right=0.95, left=0.2, bottom=0.2, top=0.75)

            param = "logt_heat"

            fid_name = "joint_fid"

            fid_CI = chains[fid_name].chain_stats["M-HPDI"][param]
            fid_MAP = chains[fid_name].chain_stats["marg MAP"][param]

            # ax.axvspan(7.6, 8.0, facecolor="grey", alpha=0.5)
            ax.axvline(7.6, c="grey",)
            ax.axvline(8.0, c="grey")
            ax.axvspan(*fid_CI, facecolor="C2", alpha=0.5)
            ax.axvline(fid_MAP, color="C2", ls="--")

            chain_selection = [
                                {"name" : "EE_fid",},
                                {"name" : "TE_fid",},
                                {"name" : "joint_fid",},
                                # {"name" : "TE_no_y_IA",},
                                # {"name" : "joint_no_y_IA",},
            ]
            chains_to_plot = select_chains(chains, chain_selection)
            chain_labels = ["Cosmic shear", "shear--tSZ", "Cosmic shear + shear--tSZ", 
                            #  "shear--tSZ, no IA--tSZ", 
                            #  "Cosmic shear + shear--tSZ, no IA--tSZ"
                             ]

            for c, label in zip(chains_to_plot, chain_labels):
                ls = ":" if "no IA" in label else "-"
                plot_marginal_density(c, param, ax, {"ls": ls, "label": label})

            ax.set_ylim(bottom=0)
            ax.set_xlim(7.1, 8.5)

            ax.set_ylabel(r"$P\left(\log_{10}\frac{T_\mathrm{AGN}}{\mathrm{K}}\right)$")
            ax.legend(frameon=False, loc=2, bbox_to_anchor=(0,1.5), fontsize=legend_fontsize)
            ax.set_xlabel(r"$\log_{10}\frac{T_\mathrm{AGN}}{\mathrm{K}}$")

            fig.savefig("plots/logt_heat_plot.pdf")


        if "B-modes" in make_plots:
            print("Plotting B-modes")
            Cl = {"BB": np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/data/Cl_BB_shear_KiDS1000_gal.txt")[:, 1:],
                  "TB": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/data/Cl_TB_shear_KiDS1000_gal_y_milca.txt")[:, 1:]}

            Cl_err = {"BB": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/cov/covariance_total_gaussian_BBBB.txt"))).reshape(-1, n_ell_bin).T,
                      "TB": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/cov/covariance_total_gaussian_TBTB.txt"))).reshape(-1, n_ell_bin).T,
                    }
            plotting_utils.plot_joint(
                Cls_EE=[{"name"        : "pseudo-$C_\\ell$ $B$-modes",
                        "X"           : ell_eff,
                        "Y"           : Cl["BB"], 
                        "Y_error"     : Cl_err["BB"],
                        "plot_kwargs" : {"c": "C0", "ls": "none", "marker": "."}},
                    ],
                Cls_TE=[{"name"        : "TB MILCA",
                        "X"           : ell_eff,
                        "Y"           : Cl["TB"],
                        "Y_error"     : Cl_err["TB"],
                        "plot_kwargs" : {"c": "C0", "ls": "none", "marker": "."}},
                    ],
                n_z_bin=5,
                figsize=(text_width, 0.9*text_width),
                scaling_EE=lambda x: x, scaling_TE=lambda x: x**2/(2*np.pi),
                x_data_range=(100, 1500), x_range=(100, 1800), sharey=True,
                y_range_EE=(-4e-7, 4e-7), y_range_TE=(-1.6e-9, 1.6e-9),
                y_label_EE=r"$\ell C_\ell^{\gamma_B\gamma_B}$", y_label_TE=r"$\ell^2/2\pi\ C_\ell^{\gamma_B y}$",
                legend_fontsize=legend_fontsize, label_fontsize=legend_fontsize,
                filename="plots/data_vectors_B_mode.pdf"
            )

        if "bp_vs_pCl" in make_plots:
            print("Plotting bp_vs_pCl")

            Cl = {"EE": np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/data/Cl_EE_shear_KiDS1000_gal.txt")[:, 1:],
                "TE": np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/data/Cl_TE_shear_KiDS1000_gal_y_milca.txt")[:, 1:],
                }

            Cl_err = {"EE": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/cov/covariance_total_SSC_mask_EEEE.txt"))).reshape(-1, n_ell_bin).T,
                    "TE": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_y_milca/likelihood/cov/covariance_total_SSC_mask_TETE.txt"))).reshape(-1, n_ell_bin).T,
                    }

            n_ell_bin_bp = 8

            Cl_bp = {"EE": np.zeros((n_ell_bin_bp, len(field_idx_EE))),
                    "BB": np.zeros((n_ell_bin_bp, len(field_idx_EE)))}
            Cl_err_bp = {"EE": np.zeros((n_ell_bin_bp, len(field_idx_EE))),
                        "BB": np.zeros((n_ell_bin_bp, len(field_idx_EE)))}
            ell_bp = {}

            for i, idx in enumerate(field_idx_EE):
                (ell_bp["EE"], Cl_bp["EE"][:, i], Cl_err_bp["EE"][:, i],
                Cl_bp["BB"][:, i], Cl_err_bp["BB"][:, i]) = np.loadtxt("../../../KiDS/Cat_to_Obs_K1000_P1/data/Data_Plots/"
                                            "Pkk/Pkk_data/xi2bandpow_output_K1000_ALL_BLIND_C_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_goldclasses_Flag_SOM_Fid_nbins_8_Ell_100.0_1500.0_"
                                            f"zbins_{idx[1]+1}_{idx[0]+1}.dat",
                                            unpack=True)
                Cl_bp["EE"][:, i] = Cl_bp["EE"][:, i]/ell_bp["EE"]**2
                Cl_err_bp["EE"][:, i] = Cl_err_bp["EE"][:, i]/ell_bp["EE"]**2
                Cl_bp["BB"][:, i] = Cl_bp["BB"][:, i]/ell_bp["EE"]**2
                Cl_err_bp["BB"][:, i] = Cl_err_bp["BB"][:, i]/ell_bp["EE"]**2
            ell_bp["BB"] = ell_bp["EE"]

            Cl_err_bp_fits = {"EE": np.sqrt(np.diag(np.loadtxt("../proto_runs/cosmic_shear_run6_bp_cov_multinest/data/like/covariance_bp_G_cNG_SSC_m_EE.txt"))).reshape(-1, n_ell_bin)[:,2:-2].T}

            plotting_utils.plot_cosmic_shear(
                Cls=[{"name"        : "Asgari et al. (2021) bandpowers",
                    "X"           : ell_bp["EE"],
                    "Y"           : Cl_bp["EE"], 
                    "Y_error"     : Cl_err_bp["EE"],
                    "plot_kwargs" : {"c": "C0", "ls": "none", "marker": "."}},
                    {"name"        : "pseudo-$C_\\ell$",
                    "X"           : ell_eff,
                    "Y"           : Cl["EE"], 
                    "Y_error"     : Cl_err["EE"],
                    "plot_kwargs" : {"c": "C1", "ls": "none", "marker": "."}},
                    ],
                n_z_bin=5,
                figsize=(text_width, 0.8*text_width),
                scaling=lambda x: x, x_offset=lambda x, i: x*1.1**i,
                x_data_range=(100, 1500), x_range=(100, 1800), sharey=True,
                y_range=(-4e-7, 9e-7), 
                y_label=r"$\ell C_\ell^{\gamma\gamma}$", 
                legend_fontsize=legend_fontsize, label_fontsize=legend_fontsize,
                filename="plots/data_vectors_bp_vs_pCl.pdf"
            )

        if "bp_vs_pCl_window_functions" in make_plots:
            print("Plotting bp_vs_pCl_window_functions")
            W_EE_bp = np.load("../proto_runs/cosmic_shear_run7_bp_multinest/data/like/BP_apo_8_bin_ell_100_1500_op.npy")[0, :, 0]

            W_EE_pCl = {}
            for idx in field_idx_EE:
                W_EE_pCl[idx] = np.load(f"../results/measurements/shear_KiDS1000_shear_KiDS1000/data/pymaster_bandpower_windows_{idx[0]}-{idx[1]}.npy")[0, :, 0]

            bp_bin_edges = np.geomspace(100, 1500, 9)
            bp_bin_center = np.sqrt(bp_bin_edges[:-1]*bp_bin_edges[1:])

            fig, ax = plt.subplots(8, 1, sharex=True, figsize=(column_width, 1.5*column_width))
            fig.subplots_adjust(hspace=0, top=0.99, bottom=0.1, left=0.2, right=0.99)

            for bin_idx in range(8):
                ax[bin_idx].axvspan(bp_bin_edges[bin_idx], bp_bin_edges[bin_idx+1], facecolor="grey", alpha=0.5)
                ell = np.arange(W_EE_bp.shape[1])
                ax[bin_idx].plot(ell, W_EE_bp[bin_idx]/bp_bin_center[bin_idx]**2, c=f"C0", label="Bandpowers")

                ell = np.arange(W_EE_pCl[(0,0)].shape[1])
                ax[bin_idx].plot(ell, W_EE_pCl[(4,4)][bin_idx+2], c=f"C1", label="{\\sc NaMaster} pseudo-$C_\\ell$")

                ax[bin_idx].set_ylabel("$\\mathcal{F}^{\\rm EE}_{b\\ell}$")

            ax[0].legend(frameon=False, fontsize=legend_fontsize)

            ax[-1].set_xlim(left=30, right=2000)
            ax[-1].set_xscale("log")

            ax[-1].set_xlabel(r"$\ell$")

            # fig.dpi = 300
            fig.savefig("plots/bp_vs_pCl_window_functions.pdf")

        if "bp_vs_pCl_constraints_Om_S8" in make_plots:
            print("Plotting bp_vs_pCl_constraints_Om_S8")
            width = column_width
            g = getdist.plots.get_subplot_plotter(width_inch=width, scaling=False,
                                                settings=copy.deepcopy(plot_settings))

            g.make_figure(nx=1, ny=2, ystretch=0.6, sharex=True, sharey=True)
            ax = []

            chain_selection = [
                                {"name" : "EE_bp",},
                                {"name" : "EE_hmcode_old_m"},
                            ]
            chains_to_plot = select_chains(chains, chain_selection)
            chain_labels = [get_chain_label(c) for c in chains_to_plot]

            print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")

            ax.append(g._subplot(0, 0))
            g.plot_2d(chains_to_plot, param1="omegam", param2="s8", 
                    filled=True,
                    add_legend_proxy=True,
                    colors=["C0", "C1", "C2", "C3", "C4"],
                    ax=ax[-1]
            )
            ax[-1].legend(g.contours_added[-len(chains_to_plot):],
                        ["Asgari et al. 2021 bandpowers", "Pseudo-$C_\\ell$"],
                        loc="upper right", 
                        frameon=False, fontsize=legend_fontsize)
            g._no_x_ticklabels(ax[-1])

            chain_selection = [
                                {"name" : "EE_fid"},
                                {"name" : "EE_hmcode2020_old_m"},
                                {"name" : "EE_hmcode_old_m"},
                            ]
            chains_to_plot = select_chains(chains, chain_selection)
            chain_labels = [get_chain_label(c) for c in chains_to_plot]

            print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")

            ax.append(g._subplot(0, 1, sharex=ax[0], sharey=ax[0]))
            g.plot_2d(chains_to_plot, param1="omegam", param2="s8", 
                    filled=True,
                    do_ylabel=True,
                    colors=["C3", "C4", "C9"],
                    add_legend_proxy=True,
                    ax=ax[-1]
            )
            ax[-1].legend(g.contours_added[-len(chains_to_plot):],
                        ["{\\sc HMx}",
                        "{\\sc HMCode-2020}",
                        "{\\sc HMCode-2016}"],
                        loc="upper right", 
                        frameon=False, fontsize=legend_fontsize)

            g._subplots_adjust()
            g.gridspec.update(hspace=0)
            g.export("plots/bp_vs_pCl_S8_Om.pdf")

        if "bp_vs_pCl_constraints_S8" in make_plots:
            print("Plotting bp_vs_pCl_constraints_S8")
            fig, ax = plt.subplots(1, 1, figsize=(column_width, 0.6*column_width))
            fig.subplots_adjust(top=0.98, left=0.02, right=0.98, bottom=0.2)

            chains_to_plot = [chains[n] for n in ["EE_bp", 
                                                  "EE_hmcode_old_m",
                                                  "EE_hmcode2020_old_m", 
                                                  "EE_fid"]][::-1]
            chain_labels = ["Asgari et al. (2021) bandpowers",
                            "pseudo-$C_\\ell$, {\\sc HMCode-2016}",
                            "pseudo-$C_\\ell$, {\\sc HMCode-2020}",
                            "pseudo-$C_\\ell$, {\\sc HMx}"][::-1]
            chain_colors = ["C0", "C1", "C4", "C3"][::-1]

            fid_chain = "EE_bp"
            param = "s8"

            fid_CI = chains[fid_chain].chain_stats["M-HPDI"][param]
            fid_MAP = chains[fid_chain].chain_stats["marg MAP"][param]

            # ax.axvspan(*fid_CI, facecolor="C0", alpha=0.5)
            ax.axvline(fid_CI[0], color="C0", lw=1, alpha=0.5)
            ax.axvline(fid_CI[1], color="C0", lw=1, alpha=0.5)
            ax.axvline(fid_MAP, color="C0", ls=":", lw=1)

            violin_stats = [get_violin_stats(c, param) for c in chains_to_plot]

            violin_plots = ax.violin(violin_stats, vert=False, showmeans=False, showextrema=False)

            for i, chain in enumerate(chains_to_plot):
                violin_plots["bodies"][i].set_facecolor(chain_colors[i])

                MAP = chain.chain_stats["marg MAP"][param]
                l, u = chain.chain_stats["M-HPDI"][param]
                err = np.array([MAP-l, u-MAP])[:, None]
                ax.errorbar(x=MAP, y=i+1, xerr=err, marker=".",  c=chain_colors[i], capsize=3, capthick=1)
                ax.text(0.42, i+1, chain_labels[i], fontsize=legend_fontsize)

            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_xlabel(r"$S_8$", x=0.8)
            ax.set_xlim(0.41, 0.83)
            ax.set_xticks([0.7, 0.75, 0.8])
            ax.set_xticklabels([0.7, 0.75, 0.8])

            fig.savefig("plots/bp_vs_pCl_violin_S8.pdf")
        
        if "fid_constraints_all" in make_plots:
            print("Plotting fid_constraints_all")
            chain_selection = [
                                {"name" : "EE_fid",},
                                {"name" : "TE_fid",},
                                {"name" : "joint_fid",},
                  ]
            chains_to_plot = select_chains(chains, chain_selection)

            chain_colors = ["C0", "C1", "C2"]
            chain_labels = ["Cosmic shear", "shear--tSZ", "Cosmic shear + shear--tSZ"]

            print(f"Plotting {len(chains_to_plot)} chains ({', '.join(chain_labels)})")

            # params_to_plot = ["omegam", "sigma8", "s8", "Sigmaalpha", "a_ia", "logt_heat"]
            # "Sigma0p25", "Sigma0p2", 
            params_to_plot = ["omegach2", "omegabh2", "s8", "h", "ns", "a_ia", "logt_heat", "alpha_cib", "delta_z1", "delta_z2", "delta_z3", "delta_z4", "delta_z5"]

            # params_to_plot = ["omegam", "sigma8", "s8", "h", "logt_heat", "Sigma0p25", "Tau2"]# "alpha_cib"]
            # params_to_plot = ["omegam", "sigma8", "s8", "omegabh2", "ns", "logt_heat",]# "alpha_cib"]

            print(f"Plotting parameters {' '.join(params_to_plot)}")

            width = text_width
            g = getdist.plots.get_subplot_plotter(width_inch=width, scaling=False,
                                                settings=copy.deepcopy(plot_settings))
            g.settings.legend_fontsize = 10

            g.triangle_plot(chains_to_plot,
                            params=params_to_plot,
                            filled_compare=True,
                            contour_colors=chain_colors,
                            legend_labels=chain_labels,
                            diag1d_kwargs={"normalized" : True},
                            param_limits={"s8" : (0.65, 0.85),
                                        "Sigmaalpha" : (0.65, 0.85),
                                        # "a_ia" : (-1, 2)
                                        }
                        )

            idx = params_to_plot.index("alpha_cib")
            g.subplots[-1,idx].xaxis.labelpad = 20
            g.subplots[idx, 0].yaxis.labelpad = 15

            idx = params_to_plot.index("logt_heat")
            g.subplots[idx, 0].yaxis.labelpad = 15

            g.export("plots/KiDSxPlanck_all.pdf")
        
        if "ACT_Cls" in make_plots:
            print("Plotting ACT Cls")
            Cl_ACT = {"nocib": np.loadtxt("../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocib/likelihood/data/Cl_TE_shear_KiDS1000_cel_ACT_BN_nocib.txt")[:, 1:],
                      "nocmb": np.loadtxt("../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocmb/likelihood/data/Cl_TE_shear_KiDS1000_cel_ACT_BN_nocmb.txt")[:, 1:],
                      "all": np.loadtxt("../results/measurements/shear_KiDS1000_cel_y_ACT_BN/likelihood/data/Cl_TE_shear_KiDS1000_cel_ACT_BN.txt")[:, 1:]}
            Cl_ACT_err = {"nocib": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocib/likelihood/cov/covariance_total_SSC_mask_TETE.txt"))).reshape(-1, n_ell_bin).T,
            "nocmb": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocmb/likelihood/cov/covariance_total_SSC_mask_TETE.txt"))).reshape(-1, n_ell_bin).T,
            "all": np.sqrt(np.diag(np.loadtxt("../results/measurements/shear_KiDS1000_cel_y_ACT_BN/likelihood/cov/covariance_total_SSC_mask_TETE.txt"))).reshape(-1, n_ell_bin).T}

            common_kwargs = {}#{"markersize": 3, "elinewidth": 1"}
            fig, ax = plotting_utils.plot_xcorr(
                Cls=[{"name"        : "No deprojection",
                    "X"           : ell_eff,
                    "Y"           : Cl_ACT["all"],
                    "Y_error"     : Cl_ACT_err["all"],
                    "plot_kwargs" : {"c": "C7", "ls": "none", "marker": ".", **common_kwargs}},
                    {"name"        : "CMB deprojected",
                    "X"           : ell_eff,
                    "Y"           : Cl_ACT["nocmb"],
                    "Y_error"     : Cl_ACT_err["nocmb"],
                    "plot_kwargs" : {"c": "C8", "ls": "none", "marker": ".", **common_kwargs}},
                    {"name"        : "CIB deprojected",
                    "X"           : ell_eff,
                    "Y"           : Cl_ACT["nocib"],
                    "Y_error"     : Cl_ACT_err["nocib"],
                    "plot_kwargs" : {"c": "C9", "ls": "none", "marker": ".", **common_kwargs}},
                    ],
                n_z_bin=5,
                figsize=(column_width, 0.7*column_width),
                scaling=lambda x: x**2/(2*np.pi),
                x_data_range=(100, 1500), x_range=(100, 1800), sharey=True,
                x_offset=lambda x, i: x*1.06**i,
                y_range=(-3e-9, 6.1e-9),
                y_label=r"$\ell^2/2\pi\ C_\ell^{\gamma y}$",
                field="$y$",
                legend_fontsize=legend_fontsize, label_fontsize=legend_fontsize,
            )
            ax[0,2].get_xticklabels()[1].set_visible(False)
            fig.savefig("plots/data_vectors_ACT.pdf")



    if args.make_stats:
        ################################################
        ################# Statistics ###################
        ################################################

        with open("stats/1d_tension.txt", "w") as f:
            calculate_1d_tension([
                            (chains["EE_fid"], chains["Planck_fiducial"], {"stats": ["T"]}),
                            (chains["TE_fid"], chains["Planck_fiducial"], {"stats": ["T"], "params": ["s8", "sigma8", "Sigma0p2"]}),
                            (chains["joint_fid"], chains["Planck_fiducial"], {"stats": ["T"], "params": ["s8", "sigma8", "omegam", "Sigma0p2"]}),
                            ], file=f)

        with open("stats/chain_stats.txt", "w") as f:
            for c in [
                    chains["EE_fid"],
                    chains["TE_fid"],
                    chains["joint_fid"],

                    chains["joint_no_y_IA"],
                    chains["TE_no_y_IA"],
                    chains["TE_no_SSC"],

                    chains["y_milca_nocib_marg"],
                    chains["y_nilc_nocib_marg"],
                    chains["y_yan2019_nocib_nocib_marg"],
                    chains["y_yan2019_nocib_beta1.2_nocib_marg"],
                    chains["y_ACT_nocib_marg"],
                    chains["y_ACT_nocmb_nocib_marg"],
                    chains["y_ACT_nocib_nocib_marg"],

                    # chains["EE_bp"],
                    chains["EE_hmcode_old_m"]
                    ]:
                c.chain_stats["advanced"] = calculate_advanced_chain_stats(c, file=f)



        with open("stats/constraint_improvement.txt", "w") as f:
            for param in ["s8", "Sigmaalpha", "Sigma0p2", "sigma8", "omegam"]:
                print(file=f)
                print(param, file=f)
                for name, chain_A, chain_B in [("joint", chains["joint_fid"], chains["EE_fid"]),
                                               ("no SSC", chains["TE_no_SSC"], chains["TE_fid"])]:
                    stats_A = chain_A.chain_stats
                    stats_B = chain_B.chain_stats

                    print(name, file=f)
                    for stat in ["PJ-HPDI", "M-HPDI", "std"]:
                        ci_A = stats_A[stat][param]
                        ci_B = stats_B[stat][param]

                        print(f"  {stat:<8}: {1-(ci_A[1]-ci_A[0])/(ci_B[1]-ci_B[0]):.2f} {(ci_B[1]-ci_B[0])/(ci_A[1]-ci_A[0]):.2f}", file=f)

        
        with open("stats/chi2_table_short.tex", "w") as f:
            models_top = r"""
    \begin{tabular}{lcccc}
        \toprule
        Data             & $\chi^2_{\rm MAP}$  & $N_\mathrm{data}$ &  $N_\mathrm{dof, eff}$ & PTE  \\
        \midrule
    """
                
            models_bottom = r"""
        \bottomrule
    \end{tabular}
    """

            s = ""
            inline_s = ""
            for chain, label, tag in [(chains["EE_fid"], "pseudo-$C_\\ell$ cosmic shear", "CS"),
                                  (chains["TE_fid"], "shear--tSZ", "Xcorr"),
                                  (chains["joint_fid"], "Cosmic shear + shear--tSZ", "Joint"),]:
                chi2 = -2*(chain.chain_stats["advanced"]["MAP_loglike"])
                chi2_err = np.sqrt(chain.chain_stats["advanced"]["MAP_loglike_var"])
                n_varied = chain.chain_def["n_varied"]
                n_data = chain.chain_def["n_data"]
                n_eff = chain.chain_stats["advanced"]["n_eff"]

                ndof = n_data-n_eff
                
                pte = 1 - scipy.stats.chi2(df=ndof).cdf(chi2)

                s += label \
                + f"& ${chi2:.2f}$ "\
                    f"& ${n_data}$ "\
                    f"& ${ndof:.1f}$ "\
                    f"& ${pte:.3f}$  " + r"\\" + "\n"
                
                inline_s += r"\newcommand{\chisq" + tag + r"}{\ensuremath{" + f"{chi2:.2f}" + r"}\xspace}" + "\n"
                inline_s += r"\newcommand{\ndof" + tag + r"}{\ensuremath{" + f"{ndof:.1f}" + r"}\xspace}" + "\n"

                
                
            print(models_top, file=f)
            print(s, file=f)
            print(models_bottom, file=f)
            print("", file=f)
            print("", file=f)
            print(inline_s, file=f)


        for name, MAP_name, CI_name in [("MAP", "MAP", "PJ-HPDI"), ("marg", "marg MAP", "M-HPDI")]:
            with open(f"stats/parameter_constraints_{name}.tex", "w") as f:
                constraints = {}
                for name in ["joint_fid", "TE_fid"]:
                    chain = chains[name]
                    constraints[name] = {}
                    for p in ["omegam", "sigma8", "s8", "logt_heat", "Sigma0p2"]:
                        try:
                            MAP = chain.chain_stats[MAP_name][p]
                        except KeyError:
                            MAP =  chain.chain_stats["chain MAP"][p]
                        MAP_CI = chain.chain_stats[CI_name][p][1] - MAP, chain.chain_stats[CI_name][p][0] - MAP 

                        CI_up_str = "{" + f"+{MAP_CI[0]:.2g}" + "}"
                        CI_lo_str = "{" + f"{MAP_CI[1]:.2g}" + "}"

                        MAP_str = f"{MAP:.3g}^{CI_up_str}_{CI_lo_str}"
                        
                        constraints[name][p] = MAP_str
                    
                print(r"\newcommand{\omJoint}{\ensuremath{" + constraints["joint_fid"]["omegam"] + r"}\xspace}", file=f)
                print(r"\newcommand{\sigmaeightJoint}{\ensuremath{" + constraints["joint_fid"]["sigma8"] + r"}\xspace}", file=f)
                print(r"\newcommand{\SeightJoint}{\ensuremath{" + constraints["joint_fid"]["s8"] + r"}\xspace}", file=f)
                print(r"\newcommand{\logtheatJoint}{\ensuremath{" + constraints["joint_fid"]["logt_heat"] + r"}\xspace}", file=f)
                print("", file=f)
                print(r"\newcommand{\SigmaalphaXcorr}{\ensuremath{" + constraints["TE_fid"]["Sigma0p2"] + r"}\xspace}", file=f)

