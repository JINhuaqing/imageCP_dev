# this file contains the class to conduct simple conformal prediction 
# compare with cp_simple.py, here I remove the [jin2024(annot. at p. 4)](zotero://open-pdf/library/items/FDXAWJA3?page=4&annotation=N8WXEFQ3) in the phi_fn
# i.e.,I assume M(x) is free of theta
# Now (2024-09-16), I change the name from `cp_simple1` to `cp_semi` to reflect the fact that it is a semi-parametric method
import numpy as np
from utils.misc import load_pkl, save_pkl, _set_verbose_level, _update_params
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy.optimize import minimize, brentq
from .cp_base import CPBase
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


class CPSemi(CPBase):
    def __init__(self, ys, fs, kernel_fn=None, M=None, verbose=1):
        """ 
        - args: 
            - ys (np.array): n x * array of y, the truth 
            - fs (np.array): n x * array of f(x), i.e., Ey|x
            - kernel_fn (function): the kernel function
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

         
 

    # NOTE For test only
    def _phi_fn1(self, eps, alpha, h=1):
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
        exp2 =  (kernel_vs * self.Delta).mean(axis=0) # *
        # exp3 is in denominator, do not need it  when fit phi(x) = 0
        # small_v = 1e-10
        #exp3 =  kernel_vs.reshape(-1).mean() + small_v # 1, to avoid division by zero

        p12 = self.p12_quantity * exp2[None] # n x *
        p1 = p12.sum(axis=tuple(range(1, p12.ndim)))/eps # n
        # correct the sign of p2 (on Jul 24, 2024)
        p2 =  (self.Rs <= eps).astype(float)- 1 +  alpha 
        rv = p1 + p2
        #rv = (p1 + p2)/exp3
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
        exp2 =  (kernel_vs * self.Delta).mean(axis=0) # *
        # exp3 is in denominator, do not need it  when fit phi(x) = 0
        # small_v = 1e-10
        #exp3 =  kernel_vs.reshape(-1).mean() + small_v # 1, to avoid division by zero

        p12 = self.p12_quantity * exp2[None] # n x *
        p1 = p12.sum(axis=tuple(range(1, p12.ndim)))/eps # n
        # correct the sign of p2 (on Jul 24, 2024)
        p2 =  (self.Rs <= eps).astype(float)- 1 +  alpha 
        rv = p1 + p2
        #rv = (p1 + p2)/exp3

        return rv

    #* test only
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
        exp20 =  (ind0* self.Delta).mean(axis=0) # *
        exp21 =  (ind1* self.Delta).mean(axis=0) # *
        exp2 = (exp21 - exp20)/h

        p12 = self.p12_quantity * exp2[None] # n x *
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
        exp20 =  (ind0* self.Delta).mean(axis=0) # *
        exp21 =  (ind1* self.Delta).mean(axis=0) # *
        exp2 = (exp21 - exp20)/h

        p12 = self.p12_quantity * exp2[None] # n x *
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
        # I remove bds, since I can use cp_naive to get a pretty good initial value
        opt_params_def = edict({
            #"bds": [1e-10, 1000],
        })

        opt_params = _update_params(opt_params, opt_params_def, logger=logger)
        def _obj(eps):
            if self.kernel_fn == "diff":
                vs = self._phi_fn_diff(eps, alpha, h)
            else:
                vs = self._phi_fn(eps, alpha, h)
            return vs.mean()

        naive_eps = self._fit_naive(alpha=alpha)
        bds = [naive_eps/2, naive_eps*2]
        can_vs = np.linspace(bds[0], bds[1], 5)
        #can_vs = np.logspace(np.log10(bds[0]), np.log10(bds[1]), 50)
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
            all_fils = [fil for fil in all_fils if CPSemi._sort_key(fil)[0][2:]==data_type]
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
            ys, fs = CPSemi._prepare_data(data_path, size, data_type, verbose=verbose)
            save_pkl(fpath, (ys, fs), verbose=verbose)
        else: 
            ys, fs = load_pkl(fpath, verbose=verbose)
        return ys, fs
        

    
    @staticmethod
    def get_base_h(fct, self, alpha=0.05, eps_naive=None): 
        """Get the h based on the simple normal reference rule
        """
        if eps_naive is None: 
            eps_naive = self._fit_naive(alpha=alpha)
        vs = self.Rs - eps_naive
        n = vs.shape[0]
        varv = np.var(vs)
        h = 1.06 * np.sqrt(varv) * n**(-1/5) * fct
        return h




