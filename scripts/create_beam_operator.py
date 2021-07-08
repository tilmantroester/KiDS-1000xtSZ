import sys
sys.path.append("../tools")

import numpy as np

from misc_utils import create_beam_operator, file_header


if __name__ == "__main__":
    ell = np.arange(3*2048)

    beam_Planck = 10.0
    beam_ACT = 1.6

    beam_Planck_operator = create_beam_operator(ell=ell, fwhm=beam_Planck)
    beam_ACT_operator = create_beam_operator(ell=ell, fwhm=beam_ACT)

    beam_Planck2ACT_operator = create_beam_operator(ell=ell,
                                                    fwhm_map=beam_Planck,
                                                    fwhm_target=beam_ACT)

    header_string = file_header("10' beam (Planck tSZ milca and nilc) "
                                "for nside=2048")
    np.savetxt("../data/xcorr/beams/beam_Planck.txt",
               np.diag(beam_Planck_operator),
               header=header_string)

    header_string = file_header("1.6' beam (ACT tSZ) for nside=2048")
    np.savetxt("../data/xcorr/beams/beam_ACT.txt",
               np.diag(beam_Planck2ACT_operator),
               header=header_string)

    header_string = file_header("10'->1.6' beam for nside=2048")
    np.savetxt("../data/xcorr/beams/beam_Planck2ACT.txt",
               np.diag(beam_Planck2ACT_operator),
               header=header_string)
