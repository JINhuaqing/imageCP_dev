import numpy as np
from utils.misc import load_pkl, save_pkl, _set_verbose_level, _update_params
from pathlib import Path
from tqdm import tqdm
import numpy as np
import numpy.random as npr

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -  %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 


class MyDataGen():
    def __init__(self, 
                 xysize, 
                 A, 
                 X_type='normal',
                 noise_type='normal',
                 noise_std=1,
                 verbose=1) -> None:
        """Initialize the class
        Generate data (X, Y), where Y = AX + noise
        args: 
            - xysize (tuple): size of the data, e.g., (100, 100)
            - A (np.array): matrix, sizey x sizex
            - X_type (str): type of the data, 'normal' or 'binary'
            - noise_type (str): type of the noise, 'normal' or 'uniform' or "t"
            - noise_std (float): standard deviation of the noise
            - verbose (int): verbose level, the larger the more verbose
        """
        _set_verbose_level(verbose=verbose, logger=logger)
        self.verbose = verbose
        self.xysize = xysize
        self.A = A 
        self.X_type = X_type.lower()
        self.noise_type = noise_type.lower()
        self.noise_std = noise_std


    @staticmethod
    def gen_normalX(n, size):
        """generate n samples of data from normal distribution
        args: 
            - n (int): number of samples to generate
            - size (int): size of the data, 
        """
        X = npr.randn(n, size)  # n x sizex 
        return X
    
    @staticmethod
    def gen_binaryX(n, size, p=0.5, numcls=2):
        """generate n samples of data from categorical distribution
        args: 
            - n (int): number of samples to generate
            - size (int): size of the data, 
        """
        X = npr.choice(numcls, size=(n, size), p=[1-p, p])  # n x sizex
        return X

    @staticmethod
    def gen_YcX(X, A, noise_std, noise_type='normal'):
        """generate Y = AX + noise
        args: 
            - X (np.array): data, n x sizex
            - A (np.array): matrix, sizey x sizex
            - noise_std (float): standard deviation of the noise
            - noise_type (str): type of the noise, 'normal' or 'uniform' or "t"
        """
        Yraw = X @ A.T
        if noise_type == 'normal':
            noise = npr.randn(*Yraw.shape)
        elif noise_type == 'uniform':
            noise = npr.uniform(-1, 1, size=Yraw.shape)
        elif noise_type == "lognormal":
            noise = npr.lognormal(0, 1, size=Yraw.shape)
            noise = noise - np.exp(0.5)
        elif noise_type == 't':
            noise = npr.standard_t(3, size=Yraw.shape)
        noise = noise / np.std(noise, axis=0)[None] * noise_std
        Y = Yraw + noise
        return Y, noise

    
    def __call__(self, n, seed=None):
        """Generate n samples of data
        args: 
            - n (int): number of samples to generate
        """
        np.random.seed(seed)
        n = int(n)
        if self.X_type.startswith("normal"):
            X = self.gen_normalX(n, self.xysize[0])
        else:
            X = self.gen_binaryX(n, self.xysize[0])
        Y, noise = self.gen_YcX(X, 
                         self.A, 
                         noise_std=self.noise_std, 
                         noise_type=self.noise_type)
        if Y.shape[1] == 1:
            Y = Y.flatten()
        return X, Y