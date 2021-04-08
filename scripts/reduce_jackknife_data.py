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
    
    jk_data = {}
    for d in dirs:
        jk_dirs = glob.glob(os.path.join(d, f"{args.prefix}_*_*"))
        name = os.path.split(d)[1]
        print(f"Reducing {name} ({len(jk_dirs)} entries)")
        jk_data[name] = {}
        for jk_dir in jk_dirs:
            jk_name = os.path.split(jk_dir)[1]
            block_key, block_idx = jk_name.split("_")[-2:]
            block_idx = int(block_idx)

            data = np.loadtxt(os.path.join(jk_dir, "spice.cl"),  usecols=[0,7,8])
            if block_key not in jk_data[name]: jk_data[name][block_key] = {}
            jk_data[name][block_key][int(block_idx)] = data[:,1:]

    for name in jk_data.keys():
        for block_key in jk_data[name].keys():
            indicies = sorted(list(jk_data[name][block_key].keys()))
            jk_data[name][block_key] = np.array([jk_data[name][block_key][i] for i in indicies])
            print(f"{name}, block key {block_key}: {jk_data[name][block_key].shape}")

    with open(outfile, "wb") as f:
        pickle.dump(jk_data, f)
            
            
            

