import numpy as np

import sys
import datetime

PI = np.pi


def file_header(header_info=None, script_name=None,
                author="Tilman Tröster", email="tilman@troester.space"):
    if script_name is None:
        script_name = " ".join(sys.argv)
    now = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    s = f"Created by {author} ({email}) on {now} using {script_name}"
    if header_info is not None:
        s += "\n" + header_info

    return s


def make_binning_operator(x, x_min, x_max, n_bin, weights=None,
                          binning="linear", squared=False, namaster=False):

    if binning == "linear":
        bin_edges = np.linspace(x_min, x_max, n_bin+1, endpoint=True)
    elif binning == "log":
        bin_edges = np.geomspace(x_min, x_max, n_bin+1, endpoint=True)
    else:
        raise ValueError(f"Binning type {binning} not supported.")

    w = np.ones_like(x, dtype=np.float64) if weights is None else weights

    if not namaster:
        B = np.zeros((n_bin, len(x)))
        for i in range(n_bin):
            M = np.logical_and(bin_edges[i] <= x, x < bin_edges[i+1])

            if squared:
                B[i, M] = w[M]**2/np.sum(w[M])**2
            else:
                B[i, M] = w[M]/np.sum(w[M])
        return B
    else:
        B = np.full(len(x), fill_value=-1, dtype=int)

        for i in range(n_bin):
            M = np.logical_and(bin_edges[i] <= x, x < bin_edges[i+1])
            B[M] = i
        return B


def create_beam_operator(ell, fwhm=None, fwhm_map=None, fwhm_target=None):
    if fwhm is None and fwhm_target is not None:
        fwhm_sq = fwhm_target**2 - fwhm_map**2

    sigma_sq = fwhm_sq * (1/60.0/180.0*PI/(2.0*np.sqrt(2.0*np.log(2.0))))**2
    op = np.diag(np.exp(-0.5 * ell**2 * sigma_sq))
    return op


def read_partial_map(path, fields, fill_value=None, scale=None):
    import astropy.io.fits
    import healpy

    if fill_value is None:
        fill_value = healpy.UNSEEN

    maps = []
    with astropy.io.fits.open(path) as hdu:
        for i, field in enumerate(fields):
            nside = hdu[field].header["nside"]
            m = np.full(healpy.nside2npix(nside),
                        fill_value=fill_value,
                        dtype=np.float32)
            name = hdu[field].data.names[1]
            m[hdu[field].data["pixel"]] = hdu[field].data[name]
            if scale is not None:
                m[hdu[field].data["pixel"]] *= scale[i]
            maps.append(m)

    return maps


def printflush(*args, **kwargs):
    print(*args, **kwargs, flush=True)
