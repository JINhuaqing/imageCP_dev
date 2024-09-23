# this file contains the class to select the hyperparameter h for cp_semi.py
import numpy as np
from utils.misc import load_pkl, save_pkl, _set_verbose_level, _update_params
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy.optimize import minimize, brentq
from .cp_semi import CPSemi
from itertools import product
from joblib import Parallel, delayed
import pdb

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -  %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 


class CPSemiTuneh():
    def __init__(self, 
                 ys, 
                 fs, 
                 hfcts=5,
                 kernel_fn=None, 
                 M=None, 
                 verbose=1):
        """ 
        - args: 
            - ys (np.array): n x * array of y, the truth 
            - fs (np.array): n x * array of f(x), i.e., Ey|x
            - hfcts (list-like or int): the list of h factors to be selected
                if int, then it is the number of h factors to be selected
            - kernel_fn (function): the kernel function
            - verbose (int): the verbosity level, the larger the more verbose
        """
        if isinstance(hfcts, int):
            hfcts = np.logspace(-4, 4, hfcts)

        self.hfcts = hfcts
        self.kernel_fn = kernel_fn  
        self.verbose = verbose
        self.ys = ys
        self.fs = fs
        self.M = M


    def _get_result(self, trys, trfs, 
                    teys, tefs,
                    hfct, 
                    alpha, 
                    opt_params):
        """
        Get the result for given cys, cfs, hfct, and alpha
        """
        cp = CPSemi(ys=trys, fs=trfs,
                    kernel_fn=self.kernel_fn, 
                    M=self.M, 
                    verbose=0)
        h = cp.get_base_h(hfct, cp, alpha=alpha)
        cp.fit(alpha=alpha, h=h, opt_params=opt_params)
        _, insets = cp.predict(tefs, teys)
        return cp.eps, insets


    def Boostrap(self, num_rep=100, 
                 tr_ratio=0.8,
                 alpha=0.05,
                 n_jobs=1,
                 opt_params={}):
        """Conduct the bootstrap to select the best h factor
        """
        n = self.ys.shape[0]

        def _obj_fn(idx, hfct): 
            np.random.seed(idx)
            tr_idxs = np.sort(np.random.choice(n, int(tr_ratio*n), replace=False))
            te_idxs = np.sort(np.setdiff1d(np.arange(n), tr_idxs))
            trfs, trys = self.fs[tr_idxs], self.ys[tr_idxs]
            tefs, teys = self.fs[te_idxs], self.ys[te_idxs]
            eps, insets = self._get_result(trys, trfs, 
                                           teys, tefs, 
                                           hfct, alpha, opt_params)
            return hfct, eps, insets

        pbar = product(range(num_rep), self.hfcts)
        if self.verbose >=1: 
            pbar = tqdm(pbar, total=len(self.hfcts)*num_rep, desc="Bootstraping")
        with Parallel(n_jobs=n_jobs) as parallel:
            res = parallel(delayed(_obj_fn)(idx, hfct) for idx, hfct in pbar)
        
        sum_res = self.post_analysis(res)
        # select h such that the empirical coverage rate is close to 1-alpha
        cov_idxs = np.abs(sum_res[:, -1]-(1-alpha)) <=0.02
        if cov_idxs.sum() == 0:
            logger.warning(f"Cannot find the h factor such that the empirical coverage rate is close to 1-alpha.")
            cov_idxs = np.ones(len(self.hfcts), dtype=bool)
        sel_sum_res = sum_res[cov_idxs]
        hfcts = self.hfcts[cov_idxs]
        best_idx = np.argmin(sel_sum_res[:, 1])
        opthfct = hfcts[best_idx]
        logger.info(f"Best h factor is {opthfct} at which the empirical coverage rate is {sel_sum_res[best_idx, -1]:.3f} and var of eps is {sel_sum_res[best_idx, 1]:.3E}")
        return opthfct, sum_res

        
    def post_analysis(self, res):
        """ 
        Analyze the results of the bootstrap
        args:
            - res (list): the list of results from the bootstrap
        return:
            - sum_res (np.array): the list of summary results, order is the same as self.hfcts
        """
        sum_res = []
        for hfct in self.hfcts: 
            epses = [r[1] for r in res if r[0] == hfct]
            in_sets = [r[2] for r in res if r[0] == hfct]
            cres = np.mean(epses), np.var(epses), np.mean(in_sets)
            sum_res.append(cres)
        return np.array(sum_res)
                