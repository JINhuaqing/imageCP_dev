# basic simulation to explore 
# Here I try to evaluate the var of eps and the in-set rate 
# I tune h with bootstrap
# now I change the data generation from g(AX) to A g(X) and add spline transformation before fitting the model

# A10 indicates that I use A * 10 to generate the data
# FixA indicates that I fix A for all the repetitions
PREFIX = "NewExpsqFixASplineTuneh"

import sys
from jin_utils import get_mypkg_path
pkgpath = get_mypkg_path()
sys.path.append(pkgpath)
from constants import RES_ROOT, DATA_ROOT

from easydict import EasyDict as edict
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from cp_predict import CPNaive, CPSemi, CPSemiTuneh
from utils.misc import save_pkl, num2str
from joblib import Parallel, delayed
from data_gen import MyDataGen
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline


import argparse

parser = argparse.ArgumentParser(description='Simulation')
parser.add_argument('--n_jobs', type=int, default=1, help='number of jobs')
parser.add_argument('--ntrain', type=int, default=-1, help='sample size of training')
parser.add_argument('--hfct', type=float, default=-1.0, help='h factor')
# normal or lognormal
parser.add_argument('--noise_type', type=str, default="lognormal", help='noise type')
# LIN or SVR 
parser.add_argument('--fmodel_type', type=str, default="LIN", help='fmodel type')
# none or diff
parser.add_argument('--kernel_fn', type=str, default="none", help='kernel type')
parser.add_argument('--X_type', type=str, default="normal", help='X type')

args = parser.parse_args()

config = edict()
config.alpha = 0.10
config.ntrain = args.ntrain
config.xysize = (50, 5)
config.can_hfcts = np.logspace(-4, 4, 15)
config.split_ratio = 0.5
config.fmodel_type = args.fmodel_type
config.opt_params={"bds": (0.01, 1000)}
config.M = None
config.noise_std = 0.5
config.noise_type = args.noise_type
config.X_type = args.X_type
config.model_type = "expsq"
if args.kernel_fn == "none":
    config.kernel_fn = None
else: 
    config.kernel_fn = args.kernel_fn

config.verbose = 2

# num of repetitions
config.nrep = 100

fold_name = f"{PREFIX}_X{config.X_type}_sizexy-{config.xysize[0]}x{config.xysize[1]}_noise-{num2str(config.noise_std)}_kernel-{config.kernel_fn}_fmodel-{config.fmodel_type}_noise-{config.noise_type}"
#fold_name = f"{PREFIX}_sizexy-{config.xysize[0]}x{config.xysize[1]}_noise-{num2str(config.noise_std)}"
config.res_root = RES_ROOT/fold_name

np.random.seed(423)
A = npr.randn(*config.xysize).T;
# load data to get eps 
def _gen_data(A, ntrain, config, seed):
    datagen = MyDataGen(xysize=config.xysize, 
                        A=A, 
                        model_type=config.model_type,
                        X_type=config.X_type, 
                        noise_type=config.noise_type, 
                        noise_std=config.noise_std)
    np.random.seed(seed)
    seeds = npr.randint(0, 1000000, 2)    
    Xtrain, Ytrain = datagen(n=ntrain, seed=seeds[0]);
    Xtest, Ytest = datagen(n=10000, seed=seeds[1]);
    return Xtrain, Ytrain, Xtest, Ytest

def _get_model(typ="SVR"):
    if typ == "SVR":
        model =  SVR(kernel="rbf")
    elif typ == "MLP":
        model = MLPRegressor(hidden_layer_sizes=(1000,50), max_iter=1000, alpha=1)
    elif typ == "LIN": 
        model = LinearRegression()
    
    if config.xysize[1] > 1:
        model = MultiOutputRegressor(model)
    return model
    
def _run_fn(seed, ntrain):
    # generate data
    Xtrain, Ytrain, Xtest, Ytest = _gen_data(A, ntrain, config, seed)
    res = edict()

    # non-split data
    ## fit fmodel
    fmodel = _get_model(typ=config.fmodel_type)
    # add spline transformation
    spline = SplineTransformer(degree=3, n_knots=4, include_bias=True)
    fmodel = make_pipeline(spline, fmodel);

    fmodel.fit(Xtrain, Ytrain); 
    cur_ys_train, cur_fs_train = Ytrain, fmodel.predict(Xtrain)
    cur_ys_test, cur_fs_test = Ytest, fmodel.predict(Xtest)

    ## my method
    cpfit = CPSemi(
        cur_ys_train, cur_fs_train,
                     kernel_fn=config.kernel_fn, 
                     M=config.M,
                     verbose=config.verbose)
    cpfit_naive = CPNaive(cur_ys_train, cur_fs_train,
                          M = config.M,
                          verbose=config.verbose)
    cpfit_naive.fit(alpha=config.alpha)

    ## naive conformal prediction 
    _, in_sets = cpfit_naive.predict(cur_fs_test, cur_ys_test)
    res["naive-nospl"] = {"eps": cpfit_naive.eps, "in_sets": in_sets.mean()}

    cp_cv = CPSemiTuneh(ys=cur_ys_train, 
                        fs=cur_fs_train, 
                        hfcts=config.can_hfcts, 
                        kernel_fn=config.kernel_fn, 
                        M=config.M, verbose=0)
    hfct, _ = cp_cv.Boostrap(num_rep=100, 
                             alpha=config.alpha, 
                             n_jobs=1, 
                             opt_params=config.opt_params)
    h = CPSemi.get_base_h(hfct, cpfit, eps_naive=cpfit_naive.eps, alpha=config.alpha)
    cpfit.fit(alpha=config.alpha, h=h, opt_params=config.opt_params)
    _, in_sets = cpfit.predict(cur_fs_test, cur_ys_test);
    res["wvar"] = {"eps": cpfit.eps, "in_sets": in_sets.mean()}

    # split data 
    tr_idxs = np.sort(npr.choice(ntrain, int(ntrain*config.split_ratio), replace=False))
    Xtrain1, Ytrain1 = Xtrain[tr_idxs], Ytrain[tr_idxs]
    Xtrain2, Ytrain2 = np.delete(Xtrain, tr_idxs, axis=0), np.delete(Ytrain, tr_idxs, axis=0)
    ## fit fmodel
    fmodel1 = _get_model(typ=config.fmodel_type)
    fmodel1.fit(Xtrain1, Ytrain1); 
    cur_ys_train2, cur_fs_train2 = Ytrain2, fmodel1.predict(Xtrain2)
    cur_ys_test_naive, cur_fs_test_naive = Ytest, fmodel1.predict(Xtest)

    cpfit2 = CPSemi(
        cur_ys_train2, cur_fs_train2,
                     kernel_fn=config.kernel_fn, 
                     M=config.M,
                     verbose=config.verbose)
    ## naive conformal prediction
    cpfit2_naive = CPNaive(cur_ys_train2, cur_fs_train2, 
                          M=config.M,
                          verbose=config.verbose)
    cpfit2_naive.fit(alpha=config.alpha)
    _, in_sets = cpfit2_naive.predict(cur_fs_test_naive, cur_ys_test_naive)
    res["naive"] = {"eps": cpfit2_naive.eps, "in_sets": in_sets.mean()}


    res["info"] = {"seed": seed, 
                   "hfct": hfct, "h": h, 
                   "ntrain": ntrain}
    del cpfit, cpfit2
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

if not config.res_root.exists():
    config.res_root.mkdir(parents=True, exist_ok=True)

save_pkl(config.res_root/"basic_config.pkl", config, is_force=True)
for ntrain in tqdm(ntrains, desc="size"):
    file_root = config.res_root/f"size-{num2str(ntrain)}.pkl"
    if file_root.exists():
        continue
    with Parallel(n_jobs=n_jobs) as parallel:
        ress = parallel(delayed(_run_fn)(seed, ntrain) for seed in range(config.nrep))
    save_pkl(file_root, ress)
