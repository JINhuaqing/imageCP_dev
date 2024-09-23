# this file contains the base class for cp
import stat
import numpy as np
from utils.misc import _set_verbose_level

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -  %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 


class CPBase(): 
    def __init__(self, kernel_fn=None, verbose=1):
        """ 
        - args: 
            - kernel_fn (function): the kernel function
            - verbose (int): the verbosity level, the larger the more verbose
        """
        _set_verbose_level(verbose, logger)

        if kernel_fn is None:
            kernel_fn = CPBase.gaussian_kerfn
        
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.eps = None



    @staticmethod
    def preprocess(*vss):
        nvss = []
        for vs in vss:
            if vs.shape[-1] == 1 and vs.ndim == 2:
                vs = vs.squeeze(-1)
            nvss.append(vs)
        return nvss

         
    def fit(self, alpha):
        pass
      
    def _fit_naive(self, alpha):
        """Get the critical value eps
        - args:
            - alpha (float): the significance level
        """
        ncal = len(self.Rs)
        empcov = (1 - alpha) * (1 + 1/ncal)
        # i.e., ncal > (1/alpha-1):
        if empcov > 1: 
            eps = np.max(self.Rs)
            logger.warning(f"ncal is too small to adjust the empirical coverage rate to be 1-alpha, so we use the max Rs as the critical value.")
        else: 
            eps = np.quantile(self.Rs, empcov)
        return eps

    def predict(self, fs_test, ys_test):
        """
        make prediction on the test set
        - args:
            - fs_test (np.array): n x * array of fs_test
            - ys_test (np.array): n x * array of ys_test
        - return:
            - Rs_test (np.array): n array of the non-conformity score
            - in_sets (np.array): n boolean array, whether the fs_test are in the set or not
        """
        fs_test, ys_test = CPBase.preprocess(fs_test, ys_test)
        assert self.eps is not None, "Please fit the model first"
        Delta_test = ys_test - fs_test
        Rs_test = self.R_fn(Delta_test, self.M)
        in_sets = Rs_test <= self.eps
        return Rs_test, in_sets


    @staticmethod
    def R_fn(Delta, M):
        """
        calculate the R value (i.e., non-conformity score)
        - args: 
            - Delta (np.array): n x * array of Delta
            - M (np.array): * array of M or n x * array of M
        """
        data_size = tuple(range(1, Delta.ndim))
        if Delta.ndim > M.ndim: 
            M = M[None]
        Rs = np.sqrt(np.sum(Delta**2/M, axis=data_size)) # n
        return Rs


    @staticmethod
    def gaussian_kerfn(x, h=1):
        """Gaussian kernel function.
        args: 
            - x (np.ndarray): input data
            - h (float): bandwidth
        returns: 
            - np.ndarray: kernel density estimate
        """
        return np.exp(-x**2/(2*h**2))
        #return np.exp(-x**2/(2*h**2))/(np.sqrt(2*np.pi)*h)

    @staticmethod
    def bump_kerfn(x, h=1):
        """Bump kernel function.
        from https://omni.wikiwand.com/en/articles/Bump_function
        args: 
            - x (np.ndarray): input data
            - h (float): bandwidth
        returns: 
            - np.ndarray: kernel density estimate
        """
        y = np.exp(-h**2/(h**2-x**2))
        idx0 = np.bitwise_or(x<=-h, x>=h)
        y[idx0] = 0
        y = y * np.exp(1)
        return y
