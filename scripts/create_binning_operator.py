import sys
sys.path.append("../tools")

import numpy as np

from misc_utils import make_binning_operator, file_header


if __name__ == "__main__":

    ell = np.arange(3001)

    # Gives 8 bins between 100 and 1500
    ell_min = 50.81327482
    ell_max = 2951.9845069
    n_ell_bin = 12
    bin_kind = "log"

    B_namaster = make_binning_operator(x=ell, x_min=ell_min, x_max=ell_max,
                                       n_bin=n_ell_bin, weights=(2*ell+1),
                                       binning=bin_kind, namaster=True)

    B = make_binning_operator(x=ell, x_min=ell_min, x_max=ell_max,
                              n_bin=n_ell_bin, weights=(2*ell+1),
                              binning=bin_kind)
    B_cov = make_binning_operator(x=ell, x_min=ell_min, x_max=ell_max,
                                  n_bin=n_ell_bin, weights=(2*ell+1),
                                  binning=bin_kind, squared=True)

    header_string = file_header(f"Binning operator for Cls, l=0,..., {ell[-1]}."
                                f" Cls weighted by (2*l + 1).")

    np.savetxt(f"../data/xcorr/bin_operator_{bin_kind}_n_bin_{n_ell_bin}_"
               f"ell_{ell_min:.0f}-{ell_max:.0f}.txt",
               B,
               header=header_string)

    np.savetxt(f"../data/xcorr/bin_operator_{bin_kind}_n_bin_{n_ell_bin}_"
               f"ell_{ell_min:.0f}-{ell_max:.0f}_squared_weights.txt",
               B_cov,
               header=header_string)

    np.savetxt(f"../data/xcorr/bin_operator_{bin_kind}_n_bin_{n_ell_bin}_"
               f"ell_{ell_min:.0f}-{ell_max:.0f}_namaster.txt",
               B_namaster,
               header=header_string)
