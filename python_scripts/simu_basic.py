# basic simulation to explore 
# Here I try to evaluate the var of eps and the in-set rate 

PREFIX = "BASE-traintestfull"

import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, DATA_ROOT

from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
from cp_predict.cp_simple import CPSimple
from utils.misc import save_pkl, num2str
from joblib import Parallel, delayed


import argparse

parser = argparse.ArgumentParser(description='Simulation')
parser.add_argument('--n_jobs', type=int, default=1, help='number of jobs')
parser.add_argument('--data_type', type=str, default="ALL", help='data type')
parser.add_argument('--size_fct', type=float, default=-1.0, help='size factor')
parser.add_argument('--hfct', type=float, default=-1.0, help='h factor')

args = parser.parse_args()

config = edict()
config.data_type = args.data_type
config.size = (320, 320) 

config.alpha = 0.05
config.can_hfcts = 2.0**np.arange(-5, 8, 1)


config.data_path= DATA_ROOT/"multicoil_train_mask_type-equispaced_center_fraction-4_acceleration-400";
config.data_test_path= DATA_ROOT/"multicoil_test_full_mask_type-equispaced_center_fraction-4_acceleration-400";
#config.data_test_path= DATA_ROOT/"multicoil_val_mask_type-equispaced_center_fraction-4_acceleration-400";
config.verbose = 2

# num of repetitions
config.nrep = 100

fold_name = f"{PREFIX}_{config.data_type}_{config.size[0]}-{config.size[1]}"
config.res_root = RES_ROOT/fold_name
if not config.res_root.exists():
    config.res_root.mkdir(parents=True, exist_ok=True)

# load data to get eps 
ys_train, fs_train, gs_train = CPSimple.prepare_data(data_path=config.data_path, 
                                                     size=config.size, 
                                                     data_type=config.data_type, 
                                                     verbose=config.verbose);

ys_test, fs_test, gs_test = CPSimple.prepare_data(data_path=config.data_test_path, 
                                                     size=config.size, 
                                                     data_type=config.data_type, 
                                                     verbose=config.verbose);

ntrain = len(ys_train)

def _run_fn(seed, size_fct, hfct):
    cur_ntrian = int(ntrain*size_fct)
    np.random.seed(seed)
    sel_idxs = np.sort(np.random.choice(ntrain, 
                                        cur_ntrian, 
                                        replace=False))



    cur_ys_train = ys_train[sel_idxs]
    cur_fs_train = fs_train[sel_idxs]
    cur_gs_train = gs_train[sel_idxs]

    cpfit = CPSimple(cur_ys_train, cur_fs_train, cur_gs_train, kernel_fn=None, verbose=config.verbose)

    res = edict()
    eps_naive = cpfit.fit_root_naive(alpha=config.alpha, opt_params={"bds": (0.01, 5000)})
    in_sets = cpfit.in_set(fs_test, ys_test, eps_naive);
    res["naive"] = {"eps": eps_naive, "in_sets": in_sets.mean()}

    h = CPSimple.get_base_h(hfct, cpfit, alpha=config.alpha, eps_naive=eps_naive)
    cpfit.fit_root(alpha=config.alpha, h=h, opt_params={"bds": (0.01, 5000)})
    in_sets = cpfit.in_set(fs_test, ys_test);
    res["wvar"] = {"eps": cpfit.eps, "in_sets": in_sets.mean()}

    res["info"] = {"seed": seed, "size": cur_ntrian, "hfct": hfct, "h": h}
    del cpfit
    return res


n_jobs = args.n_jobs
if args.size_fct < 0:
    size_fcts = np.linspace(0.1, 0.9, 9)
else:
    size_fcts = [args.size_fct]
if args.hfct < 0:
    can_hfcts = config.can_hfcts
else:
    can_hfcts = [args.hfct]

save_pkl(config.res_root/"basic_config.pkl", config, is_force=True)
for hfct in tqdm(can_hfcts, desc="h"):
    for size_fct in tqdm(size_fcts, desc="size"):
        file_root = config.res_root/f"size-{num2str(size_fct)}_hfct-{num2str(hfct)}.pkl"
        if file_root.exists():
            continue
        with Parallel(n_jobs=n_jobs) as parallel:
            ress = parallel(delayed(_run_fn)(seed, size_fct, hfct) for seed in range(config.nrep))
        save_pkl(file_root, ress)
