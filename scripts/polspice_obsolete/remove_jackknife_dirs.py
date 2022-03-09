import argparse
import glob
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--prefix", required=True)

    args = parser.parse_args()
    path = args.path
    dirs = glob.glob(f"{path}/*")
    
    dirs_to_remove = []
    for d in dirs:
        dirs_to_remove += [f for f in glob.glob(os.path.join(d, f"{args.prefix}_*_*")) if os.path.isdir(f)]
        
    print("Removing")
    for d in dirs_to_remove:
        print(d)

    c = input("Continue? [yes|no]")
    if c == "yes":
        print("Deleting.")
        for d in dirs_to_remove:
            shutil.rmtree(d)