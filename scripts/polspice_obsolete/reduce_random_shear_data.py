import argparse
import glob
import os
import pickle

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", required=True)

    args = parser.parse_args()
    path = args.path
    outfile = args.output
    dirs = [d for d in glob.glob(f"{path}/*") if os.path.isdir(d)]
    
    data = {}
    for d in dirs:
        rs_dirs = glob.glob(os.path.join(d, f"{args.prefix}_*"))
        name = os.path.split(d)[1]
        print(f"Reducing {name} ({len(rs_dirs)} entries)")
        data[name] = []
        for rs_dir in rs_dirs:
            rs_name = os.path.split(rs_dir)[1]
            block_idx = rs_name.split("_")[-1]
            block_idx = int(block_idx)

            cl_data = np.loadtxt(os.path.join(rs_dir, "spice.cl"),  usecols=[0,7,8])
            data[name].append(cl_data[:,1:])

    np.savez(args.output, **data)
            
            
            

