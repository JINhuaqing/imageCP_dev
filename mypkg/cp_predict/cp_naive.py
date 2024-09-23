# this file contains the class to conduct naive conformal prediction 
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


class CPNaive(CPBase):
    def __init__(self, ys, fs, M=None, verbose=1):
        """ 
        - args: 
            - ys (np.array): n x * array of y, the truth 
            - fs (np.array): n x * array of f(x), i.e., Ey|x
            - verbose (int): the verbosity level, the larger the more verbose
        """
        _set_verbose_level(verbose, logger)
        super().__init__(kernel_fn=None, verbose=verbose)
        ys, fs = self.preprocess(ys, fs)
        if M is None:
            M = np.var(ys-fs, axis=0) # *
        Delta = ys - fs  # n x * 


        self.M = M
        self.Rs = self.R_fn(Delta, M) # n

         
    def fit(self, alpha):
        """Get the critical value eps
        - args:
            - alpha (float): the significance level
        """
        self.eps = self._fit_naive(alpha=alpha)
