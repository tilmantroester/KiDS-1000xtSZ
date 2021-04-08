import numpy as np
import scipy.interpolate
pi = np.pi

from cosmosis.datablock import option_section

def setup(options):
    filter_function = options.get_string(option_section, "filter", "gaussian").lower()
    if filter_function == "gaussian":
        if options.has_value(option_section, "fwhm"):
            fwhm = options[option_section, "fwhm"]
            sigma = fwhm/60.0/180.0*pi/(2.0*np.sqrt(2.0* np.log(2.0)))
        elif options.has_value(option_section, "sigma"):
            sigma = options[option_section, "sigma"]
        else:
            raise ValueError("Either fwhm or sigma need to be specified.")
        filter_function = lambda ell: np.exp(-0.5*ell**2*sigma**2)
    elif filter_function == "healpix_window":
        nside = options[option_section, "nside"]
        #lmax = options.get_int(option_section, "lmax", nside)
        import healpy
        wT, wP = healpy.pixwin(nside=nside, pol=True)
        ell = np.arange(len(wT))
        intp = scipy.interpolate.InterpolatedUnivariateSpline(ell, wP, k=1, ext=2)
        filter_function = intp
    else:
        raise ValueError(f"Filter {filter_function} not supported.")

    sections = options.get_string(option_section, "sections", "")
    if sections != "":
        sections = [s.strip() for s in sections.split(" ")]
    else:
        sections = []

    try:
        powers = options[option_section, "powers"]
        if isinstance(powers, int) or isinstance(powers, float):
            powers = [powers]
    except:
        powers = [1,]*len(sections)

    if len(sections) != len(powers):
        raise ValueError("Number of powers doesn't match number of sections.")

    new_section_suffix = options.get_string(option_section, "new_section_suffix", "")

    return filter_function, sections, new_section_suffix, powers


def execute(block, config):
    filter_function, sections, new_section_suffix, powers = config

    new_section = new_section_suffix != ""
    
    for section, power in zip(sections, powers):
        if section not in block:
            continue
        ell = block[section, "ell"]

        if new_section:
            output_section_name = section + "_" + new_section_suffix
            block[output_section_name, "ell"] = ell
        else:
            output_section_name = section
        
        for _, key in block.keys(section):
            if key[:4] == "bin_": 
                Cl = block[section, key] * filter_function(ell)**power
                block[output_section_name, key] = Cl

    return 0

def cleanup(config):
    pass
