# basic simulation to explore 
# Here I try to evaluate the var of eps and the in-set rate 
# I use generated data
# and I mimic the procedure of simu_basic.py, fixed f and g model with a very large training size

PREFIX = "GenSVR-fixfg"

import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, DATA_ROOT

from easydict import EasyDict as edict
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from cp_predict.cp_simple import CPSimple
from utils.misc import save_pkl, num2str
from joblib import Parallel, delayed
from data_gen import MyDataGen
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


import argparse

parser = argparse.ArgumentParser(description='Simulation')
parser.add_argument('--n_jobs', type=int, default=1, help='number of jobs')
parser.add_argument('--ntrain', type=int, default=-1, help='sample size of training')
parser.add_argument('--hfct', type=float, default=-1.0, help='h factor')

args = parser.parse_args()

config = edict()
config.alpha = 0.05
config.ntrain = args.ntrain
config.xysize = (10, 5)
config.can_hfcts = np.logspace(-5, 3, 20)


config.verbose = 2

# num of repetitions
config.nrep = 100

fold_name = f"{PREFIX}_sizexy-{config.xysize[0]}x{config.xysize[1]}"
config.res_root = RES_ROOT/fold_name
if not config.res_root.exists():
    config.res_root.mkdir(parents=True, exist_ok=True)

# load data to get eps 
np.random.seed(0)
A = npr.randn(*config.xysize).T;
datagen = MyDataGen(xysize=config.xysize, A=A);

Xtrain, Ytrain = datagen(n=100000, seed=0);
Xtest, Ytest = datagen(n=10000, seed=1);

fmodel = MultiOutputRegressor(SVR(kernel="rbf"))
#fmodel = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000))
fmodel.fit(Xtrain, Ytrain); 
gmodel = MultiOutputRegressor(SVR(kernel="rbf"))
#gmodel = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000))
gmodel.fit(Xtrain, Ytrain**2);

ys_train, fs_train, gs_train = Ytrain, fmodel.predict(Xtrain), gmodel.predict(Xtrain)
ys_test, fs_test, gs_test = Ytest, fmodel.predict(Xtest), gmodel.predict(Xtest)


ntrain_total = len(ys_train)
def _run_fn(seed, ntrain, hfct):
    np.random.seed(seed)
    sel_idxs = np.sort(np.random.choice(ntrain_total, 
                                        ntrain, 
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

    res["info"] = {"seed": seed, "hfct": hfct, "h": h, "ntrain": ntrain}
    del cpfit
    return res


n_jobs = args.n_jobs
if args.ntrain < 0:
    ntrains = [100, 500, 1000, 5000, 10000]
else:
    ntrains = [args.ntrain]
if args.hfct < 0:
    can_hfcts = config.can_hfcts
else:
    can_hfcts = [args.hfct]

save_pkl(config.res_root/"basic_config.pkl", config, is_force=True)
for hfct in tqdm(can_hfcts, desc="h"):
    for ntrain in tqdm(ntrains, desc="size"):
        file_root = config.res_root/f"size-{num2str(ntrain)}_hfct-{num2str(hfct)}.pkl"
        if file_root.exists():
            continue
        with Parallel(n_jobs=n_jobs) as parallel:
            ress = parallel(delayed(_run_fn)(seed, ntrain, hfct) for seed in range(config.nrep))
        save_pkl(file_root, ress)
