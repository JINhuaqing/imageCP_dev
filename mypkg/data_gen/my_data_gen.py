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
                 model_type='linear',
                 X_type='normal',
                 noise_type='normal',
                 noise_std=1,
                 verbose=1) -> None:
        """Initialize the class
        Generate data (X, Y), where Y = AX + noise
        args: 
            - xysize (tuple): size of the data, e.g., (100, 100)
            - A (np.array): matrix, sizey x sizex
            - model_type (str): type of the model, 'linear' or 'sin' or 'log'
            - X_type (str): type of the data, 'normal' or 'binary'
            - noise_type (str): type of the noise, 'normal' or 'uniform' or "t"
            - noise_std (float): standard deviation of the noise
            - verbose (int): verbose level, the larger the more verbose
        """
        _set_verbose_level(verbose=verbose, logger=logger)
        self.verbose = verbose
        self.xysize = xysize
        self.A = A 
        self.model_type = model_type.lower()
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
    def sin_YrawcX(X, A):
        """ 
        Generate Y = sin(2pi|AX|)
        """
        tX = np.abs(X @ A.T) * 2 * np.pi
        Yraw = 2*np.sin(tX)
        return Yraw

    @staticmethod
    def _sin_YrawcX(X, A):
        """ 
        Generate Y = A sin(2pi|X|)
        """
        sinX = np.sin(np.abs(X) * 2 * np.pi)
        Yraw = sinX @ A.T
        return Yraw

    @staticmethod
    def log_YrawcX(X, A):
        """ 
        Generate Y = log(1+|AX|)
        """
        tX = np.abs(X @ A.T)
        Yraw = np.log(1+tX)
        return Yraw

    @staticmethod
    def _expsq_YrawcX(X, A):
        """ 
        Generate Y = exp(-|AX|^2)
        """
        tX = np.abs(X @ A.T)
        Yraw = 2*np.exp(-tX**2)
        return Yraw

    @staticmethod
    def expsq_YrawcX(X, A):
        """ 
        Generate Y =A exp(-|X|^2) 
        """
        expX = np.exp(-X**2)
        Yraw = expX @ A.T
        return Yraw

    @staticmethod 
    def lin_YrawcX(X, A):
        """ 
        Generate Y = AX
        """
        Yraw = X @ A.T
        return Yraw

    @staticmethod
    def gen_YcYraw(Yraw, noise_std, noise_type='normal'):
        """generate Y = Yraw + noise
        args: 
            - Yraw (np.array): raw data of Y, n x sizey
            - noise_std (float): standard deviation of the noise
            - noise_type (str): type of the noise, 'normal' or 'uniform' or "t"
        """
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
        return Y

    
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

        if self.model_type.startswith("sin"):
            Yraw = self.sin_YrawcX(X, self.A)
        elif self.model_type.startswith("log"):
            Yraw = self.log_YrawcX(X, self.A)
        elif self.model_type.startswith("expsq"):
            Yraw = self.expsq_YrawcX(X, self.A)
        else:
            Yraw = self.lin_YrawcX(X, self.A)
        Y = self.gen_YcYraw(
                         Yraw,
                         noise_std=self.noise_std, 
                         noise_type=self.noise_type)
        if Y.shape[1] == 1:
            Y = Y.flatten()
        return X, Y
