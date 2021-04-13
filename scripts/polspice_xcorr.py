import os
import argparse

import numpy as np

import parsl
from parsl.app.app import python_app, bash_app

import parsl_config

@bash_app
def run_shear_xcorr(shear_path, probe_path, output_path, 
                    tenormfileout=False, tenormfilein=None, kernelsfileout=True,
                    thetamax=None,
                    jackknife_block_file=None,
                    jackknife_block_key=None, jackknife_block_idx=None,
                    bootstrap_method=None, bootstrap_field=None,
                    randomize_shear=False,
                    tmp_dir=None,
                    n_thread=8,
                    config={}, 
                    stdout="", stderr="", outputs=[]):
    xcorr_script = "../KiDS1000_measurements.py"
    cmd = f"python {xcorr_script} --n-thread {n_thread}" \
          f" --shear-map-path={shear_path} --y-map-path={probe_path}" \
          f" --output-path={output_path}"
    if tenormfileout:
        cmd += f" --tenormfileout"
    elif tenormfilein is not None:
        cmd += f" --tenormfilein={tenormfilein}"
    if kernelsfileout:
        cmd += f" --kernelsfileout"
    if thetamax is not None:
        cmd += f" --thetamax={thetamax}"
        cmd += f" --apodizesigma={thetamax}"
    if jackknife_block_file is not None:
        cmd += f" --jackknife-block-file={jackknife_block_file}"
    if jackknife_block_key is not None:
        cmd += f" --jackknife-block-key={jackknife_block_key}"
    if jackknife_block_idx is not None:
        cmd += f" --jackknife-block-idx={jackknife_block_idx}"
    if bootstrap_method is not None:
        cmd += f" --bootstrap-method={bootstrap_method}"
    if bootstrap_field is not None:
        cmd += f" --bootstrap-field={bootstrap_field}"
    if randomize_shear:
        cmd += f" --randomize-shear"
    if tmp_dir is not None:
        cmd += f" --tmp-dir={tmp_dir}"
    return cmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--machine", default="local")
    parser.add_argument("--n-slot", default=1)
    parser.add_argument("--n-thread", default=4)
    parser.add_argument("--partition", default="all")
    parser.add_argument("--walltime")
    parser.add_argument("--shear-path-file")
    parser.add_argument("--shear-paths", nargs="+")
    parser.add_argument("--probe-path-file")
    parser.add_argument("--probe-paths", nargs="+")

    parser.add_argument("--thetamax")

    parser.add_argument("--jackknife-block-file")
    parser.add_argument("--jackknife-block-key")
    parser.add_argument("--only-jackknife", action="store_true", default=False)

    parser.add_argument("--bootstrap-method", help="Which bootstrap method to use.")
    parser.add_argument("--bootstrap-field", help="Which field to bootstrap.")
    parser.add_argument("--bootstrap-start-idx")
    parser.add_argument("--bootstrap-end-idx")

    parser.add_argument("--randomize-shear", action="store_true", default=False)

    parser.add_argument("--tmp-dir", default="jk_tmp")
    parser.add_argument("--output-path")

    args = parser.parse_args()

    base_output_path = args.output_path

    if args.shear_path_file is not None:
        with open(args.shear_path_file, "r") as f:
            shear_path = [l.rstrip() for l in f]
    elif args.shear_paths is not None:
        shear_path = args.shear_paths
    else:
        parser.error("Either --shear-path-file or --shear-paths has to be specified.")

    if args.probe_path_file is not None:
        with open(args.probe_path_file, "r") as f:
            probe_path = [l.rstrip() for l in f]
    elif args.probe_paths is not None:
        probe_path = args.probe_paths
    else:
        parser.error("Either --probe-path-file or --probe-paths has to be specified.")

    names = [f"{os.path.split(s)[1]}-{os.path.split(p)[1]}" for s in shear_path for p in probe_path]

    if args.jackknife_block_file is not None:
        jackknife_block_file = args.jackknife_block_file
        blocks = np.load(jackknife_block_file)
        if args.jackknife_block_key is not None:
            jk_block_keys = [args.jackknife_block_key]
        else:
            jk_block_keys = list(blocks.keys())
            jk_block_keys.remove("nside")
    else:
        jk_block_keys = []

    if args.bootstrap_method is not None:
        do_bootstrap = True
        bootstrap_field = args.bootstrap_field
        bootstrap_method = args.bootstrap_method
    else:
        do_bootstrap = False

    config = {}

    tmp_dir = args.tmp_dir

    machine = args.machine
    n_slot = int(args.n_slot)
    n_thread = int(args.n_thread)
    walltime = args.walltime if args.walltime is not None else "00:30:00"
    print(f"Setting up parsl for machine '{machine}'")
    # os.makedirs("./parsl", exist_ok=True)
    # os.chdir("./parsl")
    parsl_working_dir = parsl_config.setup_parsl(machine, n_slots=n_slot, n_thread=n_thread, memory=10000//n_thread, walltime=walltime, 
                                                 parsl_dir="./parsl", partition=args.partition)
    # Parsl paths (such as the log dir) are with respect to the parsl working dir.
    log_dir = "logs"
    os.makedirs(os.path.join(parsl_working_dir, log_dir), exist_ok=True)

    print("Shear paths:")
    for p in shear_path: print(p)

    print("Probe paths:")
    for p in probe_path: print(p)

    results = []
    print("Will write cross-correlations to:")
    for s in shear_path:
        for p in probe_path:

            name = os.path.split(s.rstrip("/\\"))[1] + "-" + os.path.split(p.rstrip("/\\"))[1]
            output_path = os.path.join(base_output_path, name)
            print(output_path)

            cl_file = config.get("clfile", "spice.cl")
            output_file = os.path.join(output_path, cl_file)

            if not args.only_jackknife:
                results.append(
                    run_shear_xcorr(shear_path=os.path.abspath(s), probe_path=os.path.abspath(p), output_path=os.path.abspath(output_path), 
                                    tenormfileout=True,
                                    thetamax=args.thetamax,
                                    n_thread=n_thread,
                                    stdout=os.path.join(log_dir, f"stdout_{name}.txt"), 
                                    stderr=os.path.join(log_dir, f"stderr_{name}.txt"),
                                    outputs=[os.path.abspath(output_file)]))

            if args.randomize_shear:
                randoms_indicies = range(int(args.bootstrap_start_idx), int(args.bootstrap_end_idx))
                for rs_idx in randoms_indicies:
                    rs_name = f"random_shear_{rs_idx}"
                    rs_output_path = os.path.join(base_output_path, name, rs_name)
                    print(rs_output_path)
                    cl_file = config.get("clfile", "spice.cl")
                    rs_output_file = os.path.join(rs_output_path, cl_file)

                    results.append(
                        run_shear_xcorr(shear_path=os.path.abspath(s), 
                                        probe_path=os.path.abspath(p), 
                                        output_path=os.path.abspath(rs_output_path), 
                                        tenormfileout=False, tenormfilein=os.path.abspath(os.path.join(output_path, "spice.tenorm")),
                                        thetamax=args.thetamax,
                                        jackknife_block_idx=rs_idx,
                                        randomize_shear=True,
                                        tmp_dir=os.path.abspath(tmp_dir),
                                        n_thread=n_thread,
                                        stdout=os.path.join(log_dir, f"stdout_{name}_{rs_name}.txt"), 
                                        stderr=os.path.join(log_dir, f"stderr_{name}_{rs_name}.txt"),
                                        outputs=[os.path.abspath(rs_output_file)]))
            else:
                for block_key in jk_block_keys:
                    if do_bootstrap:
                        bootstrap_indicies = range(int(args.bootstrap_start_idx), int(args.bootstrap_end_idx))
                        for bs_idx in bootstrap_indicies:
                            bs_name = f"bs_{bootstrap_field}_{block_key}_{bs_idx}"
                            bs_output_path = os.path.join(base_output_path, name, bs_name)
                            print(bs_output_path)
                            cl_file = config.get("clfile", "spice.cl")
                            bs_output_file = os.path.join(bs_output_path, cl_file)

                            results.append(
                                run_shear_xcorr(shear_path=os.path.abspath(s), 
                                                probe_path=os.path.abspath(p), 
                                                output_path=os.path.abspath(bs_output_path), 
                                                tenormfileout=False, tenormfilein=os.path.abspath(os.path.join(output_path, "spice.tenorm")),
                                                thetamax=args.thetamax,
                                                jackknife_block_file=os.path.abspath(jackknife_block_file),
                                                jackknife_block_key=block_key, jackknife_block_idx=bs_idx,
                                                bootstrap_method=bootstrap_method, bootstrap_field=bootstrap_field,
                                                tmp_dir=os.path.abspath(tmp_dir),
                                                n_thread=n_thread,
                                                stdout=os.path.join(log_dir, f"stdout_{name}_{bs_name}.txt"), 
                                                stderr=os.path.join(log_dir, f"stderr_{name}_{bs_name}.txt"),
                                                outputs=[os.path.abspath(bs_output_file)]))

                    else:
                        for block_idx in range(blocks[block_key].shape[0]):
                            jk_name = f"jk_{block_key}_{block_idx}"
                            jk_output_path = os.path.join(base_output_path, name, jk_name)
                            print(jk_output_path)
                            cl_file = config.get("clfile", "spice.cl")
                            jk_output_file = os.path.join(jk_output_path, cl_file)

                            results.append(
                                run_shear_xcorr(shear_path=os.path.abspath(s), probe_path=os.path.abspath(p), output_path=os.path.abspath(jk_output_path), 
                                                tenormfileout=False, tenormfilein=os.path.abspath(os.path.join(output_path, "spice.tenorm")),
                                                thetamax=args.thetamax,
                                                jackknife_block_file=os.path.abspath(jackknife_block_file),
                                                jackknife_block_key=block_key, jackknife_block_idx=block_idx,
                                                tmp_dir=os.path.abspath(tmp_dir),
                                                n_thread=n_thread,
                                                stdout=os.path.join(log_dir, f"stdout_{name}_{jk_name}.txt"), 
                                                stderr=os.path.join(log_dir, f"stderr_{name}_{jk_name}.txt"),
                                                outputs=[os.path.abspath(jk_output_file)]))

    for r in results:
        r.result()
    

