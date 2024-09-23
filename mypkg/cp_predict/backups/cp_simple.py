# this file contains the class to conduct simple conformal prediction 
from os import supports_bytes_environ
import numpy as np
from utils.misc import load_pkl, save_pkl, _set_verbose_level, _update_params
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy.optimize import minimize, brentq
from ..cp_base import CPBase
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


class CPSimple(CPBase):
    def __init__(self, ys, fs, gs, kernel_fn=None, M=None, verbose=1):
        """ 
        - args: 
            - ys (np.array): n x * array of y or n , the truth 
            - fs (np.array): n x * array of f(x) or n , i.e., Ey|x
            - gs (np.array): n x * array of g(x) or n, i.e., Ey^2|x
            - kernel_fn (function): the kernel function
            - verbose (int): the verbosity level, the larger the more verbose
        """
        _set_verbose_level(verbose, logger)
        super().__init__(kernel_fn=kernel_fn, verbose=verbose)
    
        ys, fs, gs = self.preprocess(ys, fs, gs)

        # get M gx - fx * fx (note that in the note it is diag(M))
        # note that M is the var of y|x, so it should be > 0 for all x
        # In fact, if we assume M is free of x, we can use y to estimate
        if M is None:
            M = np.var(ys-fs, axis=0) # *
        # get Delta y - fx 
        Delta = ys - fs  # n x * 
        

        self.ys = ys
        self.fs = fs
        self.gs = gs

        # self.eps relies on the self.h
        self.h = None

        # TODO-LONG: The following quantities will be repeated, so I should store them 
        # but in fact they contain too much memory, so in the future, I should find a better way
        self.M = M
        self.Delta = Delta
        # it is in fact the value of R in the notes
        self.Rs = self.R_fn(Delta, M) # n

        # define some constant to avoid repeated calculation
        # theoretically, (self.ys * self.ys - self.gs -2*self.fs*self.Delta) can be delta * delta - M
        #self.p11_quantity = 1/(M**2)/2 * (Delta * Delta - M)
        self.p11_quantity = 1/(M**2)/2 * (ys * ys - gs -2*fs*Delta)
        self.p12_quantity = (1/M) * Delta

         
 

    def _phi_fn(self, eps, alpha, h=1):
        """ The influence function
        - args: 
            - eps (float): the critical value 
            - alpha (float): the signicalt level
            - h (float): the bandwidth of the kernel fn
        - return: 
            - rv (np.ndarray): n vector containing phis
        """

        kernel_vs = self.kernel_fn(self.Rs-eps, h=h)
        shape = tuple([len(self.ys)] + [1]*(self.ys.ndim-1))
        # make it compatible to the shape of Delta
        kernel_vs = kernel_vs.reshape(shape) 
        exp1 =  (kernel_vs * self.Delta * self.Delta).mean(axis=0) # *
        exp2 =  (kernel_vs * self.Delta).mean(axis=0) # *
        # exp3 is in denominator, do not need it  when fit phi(x) = 0
        # small_v = 1e-10
        #exp3 =  kernel_vs.reshape(-1).mean() + small_v # 1, to avoid division by zero

        p11 = self.p11_quantity * exp1[None]  # n x *
        p12 = self.p12_quantity * exp2[None] # n x *
        p1 = (p11 + p12).sum(axis=tuple(range(1, p11.ndim)))/eps # n
        # correct the sign (on Jul 24, 2024)
        p2 = (self.Rs <= eps).astype(float)- 1 + alpha 
        rv = p1 + p2
        #rv = (p1 + p2)/exp3

        return rv

    # to be deprecated
    def _fit(self, alpha, h=1, opt_params={}):
        """Fit to get the CV eps
        - args: 
            - alpha (float): the signicalt level
            - h (float): the bandwidth of the kernel fn
            - opt_params (dict): the optimization parameters
        - return: 
        """
        opt_params_def = edict({
            "bds": (0.01, 1000),
            # the initial value of the critical value
            "x0": 0.1, 
            "method": "L-BFGS-B", 
            "options": {"disp": self.verbose>1}
        })

        opt_params = _update_params(opt_params, opt_params_def, logger=logger)
        def _obj(eps):
            vs = self._phi_fn(eps, alpha, h)
            # original it should return the root, but we want to minimize the abs sum
            return np.abs(vs.mean())

        res = minimize(_obj, x0=opt_params.x0, 
                       bounds=[opt_params.bds], 
                       method=opt_params.method,
                       options=opt_params.options)
        self.eps = res.x
        self.h = h
        assert res.success, f"Optimization failed: {res.message}"

    def fit(self, alpha, h=1, opt_params={}):
        """Fit to get the CV eps
        - args: 
            - alpha (float): the signicalt level
            - h (float): the bandwidth of the kernel fn
            - opt_params (dict): the optimization parameters
        - return: 
        """
        epss = self._fit_root(alpha, h, opt_params) 
        if len(epss) == 0:
            self.eps = 0
        else:
            # TODO: it may not be the best way to select the eps
            self.eps = np.max(epss)
        self.h = h

    def _fit_root(self, alpha, h=1, opt_params={}):
        """Fit to get the CV eps, it use the root finding method
        - args: 
            - alpha (float): the signicalt level
            - h (float): the bandwidth of the kernel fn
            - opt_params (dict): the optimization parameters
        - return: 
        """
        opt_params_def = edict({
            "bds": [1e-10, 1000],
        })

        opt_params = _update_params(opt_params, opt_params_def, logger=logger)
        if opt_params.bds[0] is None:
            opt_params.bds[0] = 1e-10
        def _obj(eps):
            vs = self._phi_fn(eps, alpha, h)
            return vs.mean()

        can_vs = np.logspace(np.log10(opt_params.bds[0]), np.log10(opt_params.bds[1]), 50)
        can_fs = np.array([_obj(can_v) for can_v in can_vs])
        signdiff = np.diff(np.sign(can_fs))
        idxs = np.where(signdiff!=0)[0]
        epss = []
        for idx in idxs: 
            eps = brentq(_obj, a=can_vs[idx], b=can_vs[idx+1])
            epss.append(eps)
        return epss


        

    #DEP to be deprecated
    def in_set(self, fs_test, ys_test, eps=None):
        """
        assess whether the fs_test are in the set or not 
        - args:
            - fs_test (np.array): n x * array of fs_test
            - ys_test (np.array): n x * array of ys_test
            - eps (float): the critical value
        - return:
            - in_sets (np.array): n boolean array, whether the fs_test are in the set or not
        """
        if eps is None:
            assert self.eps is not None, "Please fit the model first"
            eps = self.eps
        Delta_test = ys_test - fs_test
        #Rs_test = np.linalg.norm(Delta_test * 1/np.sqrt(self.M[None]), axis=(1,2), ord="fro") # n
        Rs_test = self.R_fn(Delta_test, self.M)
        in_sets = Rs_test <= eps
        return in_sets


    @staticmethod 
    def _sort_key(fil):
        """ return the sort key based on file name 
        - args: 
            - fil (Path): the path to the file
        - returns:
            - tuple: the tuple of the sort key
        """
        ps = fil.stem.split("_")
        p0 = ps[-3]
        p1 = int(ps[-2])
        p2 = int(ps[-1].split("-")[0])
        p3 = int(ps[-1].split("-")[1])
        return (p0, p1, p2, p3)

    @staticmethod
    def _prepare_data(data_path, size, data_type, verbose=False):
        """ prepare the data for the conformal prediction 
        - args:
            - data_path (str): the path to the data 
                it is the saved inference data
            - size (list,tuple): a list/tuple of two, output size of the data
            - data_type (str): the data type 
            - verbose (bool): whether to print the progress
        """
        all_fils = list(data_path.glob("fil*.pkl"))
        # filter the files based on the data type
        if data_type != "ALL":
            all_fils = [fil for fil in all_fils if CPSimple._sort_key(fil)[0][2:]==data_type]
        ys = []
        gs = []
        fs = []
        if verbose: 
            pbar = tqdm(all_fils, desc="Loading data")
        else: 
            pbar = all_fils
        for fil in pbar:
            dat = load_pkl(fil, verbose=False)
            if dat.target.shape[0]!=size[0] or dat.target.shape[1]!=size[1]:
                continue
            if verbose: 
                pbar.set_postfix({"data": fil.stem}, refresh=True)
            ys.append(dat.target)
            fs.append(dat.fx)
            gs.append(dat.gx)
        ys = np.array(ys)
        fs = np.array(fs)
        gs = np.array(gs)
        return ys, fs, gs

    @staticmethod
    def prepare_data(data_path, size, data_type="ALL", verbose=False):
        """ A wrapper for the _prepare_data method, it can save the processed data
        - args: 
            - data_path (str): the path to folder of the data 
                it is the saved inference data
            - size (list,tuple): a list/tuple of two, output size of the data
                - (320, 320), (384, 384)
            - data_type (str): the data type 
                - "FLAIR", "T1", "T2", "T1POST", "T1PRE", "ALL"
            - verbose (bool): whether to print the progress
        """
        data_type = data_type.upper()
        assert data_type in ["FLAIR", "T1", "T1POST", "T1PRE", "T2", "ALL"], "Data type be within FLAIR, T1, T2, T1POST, T1PRE, ALL"
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        size_name = str(1000 * size[0] + size[1])
        fname = f"yfgs_{data_type}_{size_name}.pkl"
        fpath = data_path/fname
        if not fpath.exists():
            ys, fs, gs = CPSimple._prepare_data(data_path, size, data_type, verbose=verbose)
            save_pkl(fpath, (ys, fs, gs), verbose=verbose)
        else: 
            ys, fs, gs = load_pkl(fpath, verbose=verbose)
        return ys, fs, gs
        

    
    @staticmethod
    def get_base_h(fct, self, eps_naive=None, alpha=0.05): 
        """Get the h based on the simple normal reference rule
        """
        vs = self.Rs - eps_naive
        n = vs.shape[0]
        varv = np.var(vs)
        h = 1.06 * np.sqrt(varv) * n**(-1/5) * fct
        return h


    # to be deprecated
    @staticmethod 
    def __select_h(cans_hs, self, alpha, yfg=None, num_fold=5, seed=1, verbose=None):
        """Select the bandwidth h
        args: 
            - cans_hs (list): the candidates of h 
            - self (CPSimple): the instance of CPSimple 
            - alpha (float): the significance level
            - yfg (tuple): the tuple of y, f, g used for the selection 
                if None, use the self.ys, self.fs, self.gs
            - num_fold (int): the number of folds for the cross validation
            - seed (int): the random seed
            - verbose (int): the verbosity level 
                if None, use the self.verbose
        return: 
            - h (float): the selected h
        """
        np.random.seed(seed)
        if yfg is None:
            ys, fs, gs = self.ys, self.fs, self.gs
        else:
            ys, fs, gs = yfg

        if verbose is None:
            verbose = self.verbose

        n = ys.shape[0]
        n_test_per_fold = n // num_fold

        idxs = np.arange(n) 
        np.random.shuffle(idxs)
        in_setss = [[] for _ in range(len(cans_hs))]
        if verbose > 1: 
            pbar = tqdm(range(num_fold), desc="Selecting h")
        else: 
            pbar = range(num_fold)
        for cv in pbar:
            idxs_test = idxs[cv*n_test_per_fold:(cv+1)*n_test_per_fold]
            idxs_train = np.setdiff1d(idxs, idxs_test)
            ys_train, fs_train, gs_train = ys[idxs_train], fs[idxs_train], gs[idxs_train]
            ys_test, fs_test, gs_test = ys[idxs_test], fs[idxs_test], gs[idxs_test]

            cpfit = CPSimple(ys_train, fs_train, gs_train, 
                             kernel_fn=self.kernel_fn, verbose=self.verbose)
            def _obj(h):
                cpfit.fit_root(alpha=alpha, h=h)
                in_sets = cpfit.in_set(fs_test, ys_test)
                return in_sets


            for idxx, h in enumerate(cans_hs):
                in_setss[idxx].append(_obj(h))

        in_setss = np.array([np.concatenate(in_sets).mean() for in_sets in in_setss])
        logger.debug(f"in_setss: {in_setss}")
        sel_idx = np.argmin(np.abs(in_setss - (1-alpha)))
        return cans_hs[sel_idx]



