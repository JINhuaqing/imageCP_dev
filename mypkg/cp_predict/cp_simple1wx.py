# this file contains the class to conduct simple conformal prediction 
# compare with cp_simple1.py,  I add x as input, because I need the expection is conditional on x
import numpy as np
from utils.misc import load_pkl, save_pkl, _set_verbose_level, _update_params
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy.optimize import minimize, brentq
from .cp_base import CPBase
import pdb
import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -  %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 


class CPSimple1wx(CPBase):
    def __init__(self, ys, fs, xs, kernel_fn=None, M=None, verbose=1):
        """ 
        - args: 
            - ys (np.array): n x * array of y, the truth 
            - fs (np.array): n x * array of f(x), i.e., Ey|x
            - xs (np.array): n x * array of x
            - kernel_fn (function): the kernel function
                if "diff", it will use the difference between the indicator function
                if None, it will use the Gaussian kernel
            - verbose (int): the verbosity level, the larger the more verbose
        """
        _set_verbose_level(verbose, logger)
        super().__init__(kernel_fn=kernel_fn, verbose=verbose)
    
        ys, fs= self.preprocess(ys, fs)

        # get M gx - fx * fx (note that in the note it is diag(M))
        # note that M is the var of y|x, so it should be > 0 for all x
        # In fact, if we assume M is free of x, we can use y to estimate
        if M is None:
            M = np.var(ys-fs, axis=0) # *
        # get Delta y - fx 
        Delta = ys - fs  # n x * 
        

        self.ys = ys
        self.fs = fs
        self.xs = xs

        # self.eps relies on the self.h
        self.h = None

        # TODO-LONG: The following quantities will be repeated, so I should store them 
        # but in fact they contain too much memory, so in the future, I should find a better way
        self.M = M
        self.Delta = Delta
        # it is in fact the value of R in the notes
        self.Rs = self.R_fn(Delta, M) # n

        # define some constant to avoid repeated calculation
        self.p12_quantity = (1/self.M) * self.Delta

         
    @staticmethod
    def condexp_x(kvs, Delta, xs):
        """Calculate the conditional expectation of Delta given x with kernel weights
        args: 
            - kvs (np.ndarray): n x n array of kernel weights
            - Delta (np.ndarray): n x * array of Delta
            - xs (np.ndarray): n x * array of x
        """
        p = xs.shape[1]
        vs = 2**np.arange(p)
        xvs = xs @ vs # n, a vector of the x values, each value is a unique value
        kvsdelta = kvs * Delta # n x *

        # the return value
        rv = np.zeros((xs.shape[0], Delta.shape[1]))
        uni_xvs = np.sort(np.unique(xvs))
        for uni_xv in uni_xvs:
            cv = kvsdelta[xvs==uni_xv].mean(axis=0)
            rv[xvs==uni_xv] = cv
        return rv

    def _phi_fn1(self, eps, alpha, h=1):
        """ The influence function
        - args: 
            - eps (float): the critical value 
            - alpha (float): the signicalt level
            - h (float): the bandwidth of the kernel fn
        - return: 
            - rv (np.ndarray): n vector containing phis
        """

        kernel_vs = self.kernel_fn(self.Rs-eps, h-h)
        shape = tuple([len(self.ys)] + [1]*(self.ys.ndim-1))
        # make it compatible to the shape of Delta
        kernel_vs = kernel_vs.reshape(shape) 
        exp2 =  self.condexp_x(kernel_vs, self.Delta, self.xs) # n x * 

        p12 = self.p12_quantity * exp2 # n x *
        p1 = p12.sum(axis=tuple(range(1, p12.ndim)))/eps # n
        p2 =  (self.Rs <= eps).astype(float)- 1 +  alpha 
        rv = p1 + p2

        return rv.mean(), p1.mean(), p2.mean()
 

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
        exp2 =  self.condexp_x(kernel_vs, self.Delta, self.xs) # n x * 

        p12 = self.p12_quantity * exp2 # n x *
        p1 = p12.sum(axis=tuple(range(1, p12.ndim)))/eps # n
        p2 =  (self.Rs <= eps).astype(float)- 1 +  alpha 
        rv = p1 + p2

        return rv


    def _phi_fn_diff1(self, eps, alpha, h=1):
        """ The influence function based on the difference 
        In original version, we use a kernel to approximate the indicator function
        Here, we use the difference between I(Rs <= eps)  and I(Rs<= eps + h)
        - args: 
            - eps (float): the critical value 
            - alpha (float): the signicalt level
            - h (float): different between the critical value
        - return: 
            - rv (np.ndarray): n vector containing phis
        """

        ind0 = (self.Rs <= eps).astype(float)
        ind1 = (self.Rs <= (eps+h)).astype(float)
        shape = tuple([len(self.ys)] + [1]*(self.ys.ndim-1))
        # make it compatible to the shape of Delta
        ind0 = ind0.reshape(shape) 
        ind1 = ind1.reshape(shape) 

        exp20 =  self.condexp_x(ind0, self.Delta, self.xs) # n x * 
        exp21 =  self.condexp_x(ind1, self.Delta, self.xs) # n x * 
        exp2 = (exp21 - exp20)/h

        p12 = self.p12_quantity * exp2 # n x *
        p1 = p12.sum(axis=tuple(range(1, p12.ndim)))/eps # n
        p2 =  (self.Rs <= eps).astype(float)- 1 +  alpha 
        rv = p1 + p2

        return rv.mean(), p1.mean(), p2.mean()

    def _phi_fn_diff(self, eps, alpha, h=1):
        """ The influence function based on the difference 
        In original version, we use a kernel to approximate the indicator function
        Here, we use the difference between I(Rs <= eps)  and I(Rs<= eps + h)
        - args: 
            - eps (float): the critical value 
            - alpha (float): the signicalt level
            - h (float): different between the critical value
        - return: 
            - rv (np.ndarray): n vector containing phis
        """

        ind0 = (self.Rs <= eps).astype(float)
        ind1 = (self.Rs <= (eps+h)).astype(float)
        shape = tuple([len(self.ys)] + [1]*(self.ys.ndim-1))
        # make it compatible to the shape of Delta
        ind0 = ind0.reshape(shape) 
        ind1 = ind1.reshape(shape) 

        exp20 =  self.condexp_x(ind0, self.Delta, self.xs) # n x * 
        exp21 =  self.condexp_x(ind1, self.Delta, self.xs) # n x * 
        exp2 = (exp21 - exp20)/h

        p12 = self.p12_quantity * exp2 # n x *
        p1 = p12.sum(axis=tuple(range(1, p12.ndim)))/eps # n
        p2 =  (self.Rs <= eps).astype(float)- 1 +  alpha 
        rv = p1 + p2

        return rv

   
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
            logger.warning("No eps found, set eps to 0")
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
            if self.kernel_fn == "diff":
                vs = self._phi_fn_diff(eps, alpha, h)
            else:
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
            all_fils = [fil for fil in all_fils if CPSimple1wx._sort_key(fil)[0][2:]==data_type]
        ys = []
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
        ys = np.array(ys)
        fs = np.array(fs)
        return ys, fs

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
        fname = f"yfs_{data_type}_{size_name}.pkl"
        fpath = data_path/fname
        if not fpath.exists():
            ys, fs = CPSimple1wx._prepare_data(data_path, size, data_type, verbose=verbose)
            save_pkl(fpath, (ys, fs), verbose=verbose)
        else: 
            ys, fs = load_pkl(fpath, verbose=verbose)
        return ys, fs
        

    
    @staticmethod
    def get_base_h(fct, self, alpha=0.05, eps_naive=None): 
        """Get the h based on the simple normal reference rule
        """
        if eps_naive is None: 
            eps_naive = self.fit_root_naive(alpha=alpha)
        vs = self.Rs - eps_naive
        n = vs.shape[0]
        varv = np.var(vs)
        h = 1.06 * np.sqrt(varv) * n**(-1/5) * fct
        return h




