import os
import argparse
import textwrap
import tempfile
import shutil
import numpy as np
import healpy

import pylenspice.pylenspice as pylenspice


def get_field(path):
    folder, filename = os.path.split(path)
    if ":" in filename:
        filename, field = filename.split(":")
        field = int(field)
    else:
        field = 0
    
    return {"file": os.path.join(folder, filename), "field": field}


def create_jackknife_mask(block_key, block_idx, block_file, output_file=None):
    blocks = np.load(block_file)
    nside = blocks["nside"]
    block_indices = blocks[block_key][block_idx]

    mask = np.ones(healpy.nside2npix(nside), dtype=bool)
    mask[block_indices] = 0

    if output_file is not None:
        output_path = os.path.dirname(output_file)
        os.makedirs(output_path, exist_ok=True)
        healpy.write_map(output_file, mask, fits_IDL=False, dtype=bool)

    return mask

def create_bootstrap_weight_map(block_key, block_file, bootstrap_type, output_file=None):
    blocks = np.load(block_file)
    nside = blocks["nside"]

    weight_map = np.ones(healpy.nside2npix(nside), dtype=np.float32)

    n_block = len(blocks[block_key])
    
    if bootstrap_type == "classic":
        w = np.zeros(n_block)
        idx = np.random.randint(n_block, size=n_block)
        unique_idx, counts = np.unique(idx, return_counts=True)
        w[unique_idx] = counts
    elif bootstrap_type == "wild":
        w = np.random.randn(n_block)
    else:
        raise ValueError(f"Bootstrap type {bootstrap_type} not supported.")

    for i, block_indices in enumerate(blocks[block_key]):
        weight_map[block_indices] = w[i]
        
    if output_file is not None:
        output_path = os.path.dirname(output_file)
        os.makedirs(output_path, exist_ok=True)
        healpy.write_map(output_file, weight_map, fits_IDL=False, dtype=np.float32)

    return weight_map

def randomize_shear_map(shear_map_path):
    k = healpy.read_map(shear_map_path, hdu=1)
    e1 = healpy.read_map(shear_map_path, hdu=2)
    e2 = healpy.read_map(shear_map_path, hdu=3)

    alpha = np.pi*np.random.rand(e1.size)
    e = np.sqrt(e1**2 + e2**2)
    e1 = np.cos(2.0*alpha)*e
    e2 = np.sin(2.0*alpha)*e

    return k, e1, e2

def apodize_mask(masks, apodize_pixel=3, output_file=None):
    mask = healpy.read_map(masks[0]) > 0
    nside = healpy.get_nside(mask)
    apodize_sigma = apodize_pixel*healpy.nside2resol(nside)

    for m in masks[1:]:
        mask = mask & m

    mask = healpy.smoothing(mask, sigma=apodize_sigma, iter=1)
    if output_file is not None:
        healpy.write_map(mask, output_file, fits_IDL=False)
    return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    default_z_min = 0.1
    default_z_max = 1.2

    default_nside = 2048
    default_shear_catalog_hdu = 2
    
    default_y_map_field = 0

    default_polspice_path = os.environ["POLSPICE"] if "POLSPICE" in os.environ else "$HOME/Codes/PolSpice_v03-07-02/bin/spice"
    default_n_thread = 8

    default_ell_max = 3000
    default_thetamax = 180.0
    default_apodizesigma = 180.0

    parser.add_argument("--dry-run", action="store_true", help="Don't run anything.")

    parser.add_argument("--shear-catalogs", nargs="+", help="List of files to shear catalogs.")
    parser.add_argument("--shear-catalog-hdu", default=default_shear_catalog_hdu, 
                        help=f"HDU of objects in shear catalog. Default {default_shear_catalog_hdu}.")
    parser.add_argument("--z-min", default=default_z_min, help=f"Lower bound on z_B. Default {default_z_min}.")
    parser.add_argument("--z-max", default=default_z_max, help=f"Upper bound on z_B. Default {default_z_max}.")

    parser.add_argument("--nside", default=default_nside, help=f"NSIDE of shear maps. Default {default_nside}.")

    parser.add_argument("--shear-map-path", 
                        help=f"Path to store shear maps.")

    
    parser.add_argument("--y-raw-map", help="Input y map. Will be transformed to triplet format.")

    parser.add_argument("--y-map-path", help="Path to y map in triplet format.")
    parser.add_argument("--y-masks", nargs="+", help="List of masks for y map.")
    parser.add_argument("--y-weight-map", help="Weight map for Compton y parameter.")
    parser.add_argument("--use-shear-footprint", action="store_true", default=False,
                        help="Use shear map footprint for y map. Saves space for small area surveys.")


    parser.add_argument("--tmp-dir", help="Directory for temporary files.")

    parser.add_argument("--jackknife-block-file", help="File with jackknife block definitions.")
    parser.add_argument("--jackknife-block-key", help="File with jackknife block definitions.")
    parser.add_argument("--jackknife-block-idx", help="File with jackknife block definitions.")

    parser.add_argument("--bootstrap-method", help="Which bootstrap method to use.")
    parser.add_argument("--bootstrap-field", help="Which field to bootstrap (shear, probe, both).")

    parser.add_argument("--randomize-shear", action="store_true", help="Randomize the shear map.")

    parser.add_argument("--apodize-mask", help="File with jackknife block definitions.")

    parser.add_argument("--polspice-path", default=default_polspice_path, 
                        help=f"Path to PolSpice executable. Default {default_polspice_path}.")
    parser.add_argument("--n-thread", default=default_n_thread, 
                        help=f"Number of OpenMP threads to use. Default {default_n_thread}.")
    
    parser.add_argument("--output-path",
                        help=f"Path to write results.")

    parser.add_argument("--ell-max", default=default_ell_max, 
                        help=f"Maximum ell of PolSpice measurement (nlmax). Default {default_ell_max}.")
    parser.add_argument("--thetamax", 
                        help=f"PolSpice thetamax parameter.")
    parser.add_argument("--apodizesigma",  
                        help=f"PolSpice apodizesigma parameter.")
    parser.add_argument("--tenormfileout", action="store_true", default=False,
                        help=f"Output PolSpice tenormfile.")
    parser.add_argument("--tenormfilein", 
                        help=f"Path to PolSpice tenormfile.")
    parser.add_argument("--kernelsfileout", action="store_true", default=False,
                        help=f"Output PolSpice kernelsfile.")

    args = parser.parse_args()

    nside = int(args.nside)
    shear_map_path = args.shear_map_path
    y_map_path = args.y_map_path
    output_path = args.output_path

    do_jackknife = args.jackknife_block_file is not None
    do_apodize_mask = args.apodize_mask is not None

    do_bootstrap = args.bootstrap_method is not None
    if do_bootstrap:
        do_jackknife = False

    do_random_shear = args.randomize_shear

    created_shear_mask = False
    created_shear_weight = False
    created_y_mask = False
    created_shear_map = False

    create_tmp_dir = False
    if do_jackknife or do_apodize_mask or do_bootstrap or do_random_shear:
        if args.tmp_dir is None:
            create_tmp_dir = True
            tmp_dir = tempfile.mkdtemp(suffix="polspice")
            print(f"Using temp dir: {tmp_dir}")
        else:
            tmp_dir = args.tmp_dir
            os.makedirs(tmp_dir, exist_ok=True)        
    
    if shear_map_path is not None:
        shear_map_file = os.path.join(shear_map_path, "triplet.fits")
        shear_mask_file = os.path.join(shear_map_path, "doublet_mask.fits")
        shear_singlet_weight_file = os.path.join(shear_map_path, "singlet_mask.fits")
        shear_weight_file = os.path.join(shear_map_path, "doublet_weight.fits")
    if y_map_path is not None:
        y_mask_file = os.path.join(y_map_path, "singlet_mask.fits")
    if args.dry_run:
        print(args)

    if args.shear_catalogs is not None:
        shear_catalogs = args.shear_catalogs
        shear_catalog_hdu = int(args.shear_catalog_hdu)
        z_min = float(args.z_min)
        z_max = float(args.z_max)

        KiDS_column_names = {"x" : "ALPHA_J2000",
                             "y" : "DELTA_J2000",
                             "e1": "e1",
                             "e2": "e2",
                             "w" : "weight",
                             "z" : "Z_B"}

        KiDS_selection = [("weight", "gt", 0.0),]


        shear_output_filenames = {"triplet" :        shear_map_file,
                                  "singlet_mask" :   shear_singlet_weight_file,
                                  "doublet_mask" :   shear_mask_file,
                                  "doublet_weight" : shear_weight_file}

        if args.dry_run:
            print(textwrap.dedent(f"""
                                      shear_catalogs: {shear_catalogs}
                                      shear_output_filenames: {shear_output_filenames}
                                      KiDS_column_names: {KiDS_column_names}
                                      KiDS_selection: {KiDS_selection}"""))
        else:
            print("Creating shear maps.")
            pylenspice.create_shear_healpix_triplet(shear_catalogs=shear_catalogs, out_filenames=shear_output_filenames, 
                                                    hdu_idx=shear_catalog_hdu, nside=nside, flip_e1=True, convert_to_galactic=True,
                                                    partial_maps=True, 
                                                    c_correction="data", m_correction=None, column_names=KiDS_column_names, 
                                                    selections=KiDS_selection,
                                                    z_min=z_min, z_max=z_max)
    
    if args.y_raw_map is not None:
        if y_map_path is None:
            parser.error("--y-raw-map requires --y-map-path to be set.")

        y_map = get_field(args.y_raw_map)
        
        y_masks = args.y_masks
        if y_masks is not None:
            y_masks = [get_field(m) for m in y_masks]

        if args.use_shear_footprint:
            if shear_map_path is None:
                parser.error("--use-shear-footprint requires --shear-map-path to be set.")
            shear_footprint_file = os.path.join(shear_map_path, "doublet_mask.fits")
        else:
            shear_footprint_file = None

        output_filenames = { "triplet"      : os.path.join(y_map_path, "triplet.fits"),
                             "doublet_mask" : os.path.join(y_map_path, "doublet_mask.fits"),
                             "singlet_mask" : y_mask_file}

        if args.dry_run:
            print(textwrap.dedent(f"""
                                      y_maps: {y_map}
                                      y_masks: {y_masks}
                                      output_filenames : {output_filenames}
                                      shear_footprint_file : {shear_footprint_file}"""))
        else:
            print("Creating y maps.")
            pylenspice.create_foreground_healpix_triplet(y_map, y_masks, output_filenames, nside, 
                                                         coord_in="G", coord_out="G", footprint_file=shear_footprint_file
                                                        )

    if do_jackknife:
        block_key = args.jackknife_block_key
        block_idx = args.jackknife_block_idx
        if block_key is None or block_idx is None:
            parser.error("--jackknife-block-file requires --jackknife-block-key and jackknife-block-idx to be set.")
        
        jk_mask = create_jackknife_mask(str(block_key), int(block_idx), args.jackknife_block_file)

        shear_mask = healpy.read_map(shear_mask_file) > 0
        shear_mask = shear_mask & jk_mask

        shear_mask_file = os.path.join(tmp_dir, f"shear_mask_jk_{block_key}_{block_idx}.fits")
        healpy.write_map(shear_mask_file, shear_mask, dtype=np.float32, fits_IDL=False, overwrite=True)
        created_shear_mask = True

        y_mask = healpy.read_map(y_mask_file) > 0
        y_mask = y_mask & jk_mask

        y_mask_file = os.path.join(tmp_dir, f"probe_mask_jk_{block_key}_{block_idx}.fits")
        healpy.write_map(y_mask_file, y_mask, dtype=np.float32, fits_IDL=False, overwrite=True)
        created_y_mask = True

    if do_bootstrap:
        block_key = args.jackknife_block_key
        block_idx = args.jackknife_block_idx
        if block_key is None or block_idx is None:
            parser.error("--jackknife-block-file requires --jackknife-block-key and jackknife-block-idx to be set.")
        
        bs_weight_map = create_bootstrap_weight_map(block_key=str(block_key), 
                                                    block_file=args.jackknife_block_file, 
                                                    bootstrap_type=args.bootstrap_method)

        if args.bootstrap_field == "shear":
            shear_weight_map = healpy.read_map(shear_weight_file)
            m = shear_weight_map != healpy.UNSEEN
            shear_weight_map[m] = shear_weight_map[m] * bs_weight_map[m]

            shear_weight_file = os.path.join(tmp_dir, f"shear_doublet_weight_bs_{block_key}_{block_idx}.fits")
            healpy.write_map(shear_weight_file, shear_weight_map, dtype=np.float32, fits_IDL=False, overwrite=True)
            created_shear_weight = True
        else:
            raise ValueError(f"Bootstrap field {args.bootstrap_field} not supported.")

    if do_random_shear:
        block_idx = args.jackknife_block_idx
        kappa, random_e1, random_e2 = randomize_shear_map(shear_map_file)

        shear_map_file = os.path.join(tmp_dir, f"random_shear_triplet_{block_idx}.fits")
        pylenspice.write_partial_polarization_file([kappa, random_e1, random_e2], 
                                                    nside=2048, 
                                                    filename=shear_map_file, mask_value=healpy.UNSEEN, coord="G")
        created_shear_map = True


    if shear_map_path is not None and y_map_path is not None and output_path is not None:
        y_weight_map = args.y_weight_map

        polspice_path = args.polspice_path

        n_thread = args.n_thread
        
        ell_max = int(args.ell_max)

        common_params = {"nlmax"        : ell_max,
                         "decouple"     : "NO",
                         "verbosity"    : 2}

        if args.thetamax is not None:
            common_params["thetamax"] = args.thetamax
        if args.apodizesigma is not None:
            common_params["apodizesigma"] = args.apodizesigma
            
        if args.tenormfileout:
            common_params["tenormfileout"] = os.path.join(output_path, "spice.tenorm")
        elif args.tenormfilein:
            common_params["tenormfilein"] = args.tenormfilein

        if args.kernelsfileout:
            common_params["kernelsfileout"] = os.path.join(output_path, "spice.kernel")

        shear_params = {"polarization" : "YES",
                        "mapfile"      : shear_map_file,
                        "maskfile"     : shear_mask_file,
                        "maskfilep"    : shear_mask_file,
                        "weightfile"   : shear_singlet_weight_file,
                        "weightfilep"  : shear_weight_file,}

        y_params = {"mapfile2"  : os.path.join(y_map_path, "triplet.fits"),
                    "maskfile2" : y_mask_file,
                   }
        if y_weight_map is not None:
            y_params["weightfile"] = y_weight_map

        shear_y_output_params = {"clfile"        : os.path.join(output_path, "spice.cl"),}
        
        params = {**common_params,
                  **shear_params,
                  **y_params,
                  **shear_y_output_params}

        if args.dry_run:
            print(textwrap.dedent(f"""
                                      params: {params}"""))
        else:
            print("Computing cross-power spectrum.")
            os.makedirs(output_path, exist_ok=True)
            cmd = pylenspice.run_polspice(polspice_path, n_threads=n_thread, **params)
            with open(os.path.join(output_path, "polspice_cmd.txt"), "w") as f:
                f.write(cmd)

    if created_shear_mask:
        os.remove(shear_mask_file)
    if created_y_mask:
        os.remove(y_mask_file)
    if created_shear_weight:
        os.remove(shear_weight_file)
    if created_shear_map:
        os.remove(shear_map_file)

    if create_tmp_dir:
        # Remove temp dir
        print(f"Removing temp dir: {tmp_dir}")
        shutil.rmtree(tmp_dir)