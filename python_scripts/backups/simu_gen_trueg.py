# basic simulation to explore 
# *: compare with simu_gen.py, I use the true g function, but estimate f 

PREFIX = "GenTrueg-fLIN-1M1"

import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, DATA_ROOT

from easydict import EasyDict as edict
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from cp_predict import CPSimple, CPNaive, CPSimple1
from utils.misc import save_pkl, num2str
from joblib import Parallel, delayed
from data_gen import MyDataGen
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


import argparse

parser = argparse.ArgumentParser(description='Simulation')
parser.add_argument('--n_jobs', type=int, default=1, help='number of jobs')
parser.add_argument('--ntrain', type=int, default=-1, help='sample size of training')
parser.add_argument('--hfct', type=float, default=-1.0, help='h factor')

args = parser.parse_args()

config = edict()
config.alpha = 0.20
config.ntrain = args.ntrain
config.xysize = (10, 1)
config.can_hfcts = np.logspace(-4, 4, 15)
config.split_ratio = 0.5
config.fmodel_type = "LIN"
config.M = np.ones(config.xysize[1])[0]
config.noise_std = 5


config.verbose = 2

# num of repetitions
config.nrep = 100

CPSimple = CPSimple1

fold_name = f"{PREFIX}_sizexy-{config.xysize[0]}x{config.xysize[1]}_noise-{num2str(config.noise_std)}"
config.res_root = RES_ROOT/fold_name
if not config.res_root.exists():
    config.res_root.mkdir(parents=True, exist_ok=True)

# load data to get eps 

def _gen_data(ntrain, config, seed):
    A = npr.randn(*config.xysize).T;
    datagen = MyDataGen(xysize=config.xysize, A=A);
    np.random.seed(seed)
    seeds = npr.randint(0, 1000000, 2)    
    Xtrain, Ytrain = datagen(n=ntrain, seed=seeds[0]);
    Xtest, Ytest = datagen(n=10000, seed=seeds[1]);
    return Xtrain, Ytrain, Xtest, Ytest, A

def _get_gs(X, A, noise_std):
    return (X @ A.T)**2 +  noise_std**2
def _get_model(typ="SVR"):
    if typ == "SVR":
        model =  SVR(kernel="rbf")
    elif typ == "MLP":
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    elif typ == "LIN": 
        model = LinearRegression()
    
    if config.xysize[1] > 1:
        model = MultiOutputRegressor(model)
    return model


def _run_fn(seed, ntrain, hfct):
    Xtrain, Ytrain, Xtest, Ytest, A = _gen_data(ntrain, config, seed)
    fmodel = _get_model(typ=config.fmodel_type)
    fmodel.fit(Xtrain, Ytrain); 

    cur_ys_train, cur_fs_train, cur_gs_train = Ytrain, fmodel.predict(Xtrain), _get_gs(Xtrain, A, config.noise_std)
    cur_ys_test, cur_fs_test, cur_gs_test = Ytest, fmodel.predict(Xtest), _get_gs(Xtest, A, config.noise_std)

    res = edict()

    # my method
    cpfit = CPSimple(cur_ys_train, cur_fs_train, cur_gs_train, 
                     kernel_fn=None, 
                     M=config.M,
                     verbose=config.verbose)
    cpfit_naive = CPNaive(cur_ys_train, cur_fs_train, cur_gs_train, 
                          kernel_fn=None, 
                          M = config.M,
                          verbose=config.verbose)
    cpfit_naive.fit(alpha=config.alpha)
    h = CPSimple.get_base_h(hfct, cpfit, eps_naive=cpfit_naive.eps, alpha=config.alpha)
    cpfit.fit(alpha=config.alpha, h=h, opt_params={"bds": (0.01, 1000)})
    _, in_sets = cpfit.predict(cur_fs_test, cur_ys_test);
    res["wvar"] = {"eps": cpfit.eps, "in_sets": in_sets.mean()}

    # naive conformal prediction
    # split data 
    tr_idxs = np.sort(npr.choice(ntrain, int(ntrain*config.split_ratio), replace=False))
    Xtrain1, Ytrain1 = Xtrain[tr_idxs], Ytrain[tr_idxs]
    Xtrain2, Ytrain2 = np.delete(Xtrain, tr_idxs, axis=0), np.delete(Ytrain, tr_idxs, axis=0)
    fmodel1 = _get_model(typ=config.fmodel_type)
    fmodel1.fit(Xtrain1, Ytrain1); 
    cur_ys_train2, cur_fs_train2, cur_gs_train2 = Ytrain2, fmodel1.predict(Xtrain2), _get_gs(Xtrain2, A, config.noise_std)
    cur_ys_test_naive, cur_fs_test_naive, _ = Ytest, fmodel1.predict(Xtest), _get_gs(Xtest, A, config.noise_std)

    cpfit_naive = CPNaive(cur_ys_train2, cur_fs_train2, cur_gs_train2, 
                          kernel_fn=None, 
                          M=config.M,
                          verbose=config.verbose)
    cpfit_naive.fit(alpha=config.alpha)
    _, in_sets = cpfit_naive.predict(cur_fs_test_naive, cur_ys_test_naive)
    res["naive"] = {"eps": cpfit_naive.eps, "in_sets": in_sets.mean()}


    res["info"] = {"seed": seed, "hfct": hfct, "h": h, "ntrain": ntrain}
    del cpfit
    return res

n_jobs = args.n_jobs
if args.ntrain < 0:
    ntrains = [100, 500, 1000, 5000, 10000, 50000, 100000]
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
