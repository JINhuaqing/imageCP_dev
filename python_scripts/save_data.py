#!/usr/bin/env python
# coding: utf-8

# This notebook is to save the inference data, 
# 
# i.e., f(x) and g(x)
# 
# Note that they rely on the transformer param and evaluate date

# In[1]:


import sys
sys.path.append("../mypkg")
from constants import DATA_ROOT
from utils.fastmri import get_dataset, run_varnet_model, load_model
from utils.misc import save_pkl

from easydict import EasyDict as edict
import numpy as np
import torch
from tqdm import tqdm
from numbers import Number 
from joblib import Parallel, delayed
from pprint import pprint

import argparse

parser = argparse.ArgumentParser(description='Save the inference data')
parser.add_argument('--n_jobs', type=int, default=1, help='number of jobs')

args = parser.parse_args()



# In[4]:


def _get_namev(v):
    if isinstance(v, Number):
        return f"{v*100:.0f}"
    return v


# In[5]:


config = edict()
config.data_root = DATA_ROOT/"brain/multicoil_test_full"

config.mask_params = edict()
config.mask_params.mask_type = 'equispaced'
config.mask_params.center_fraction = 0.04
config.mask_params.acceleration = 4


post_fix = "_".join([f"{k}-{_get_namev(v)}" for k, v in config.mask_params.items()])
config.save_fold = DATA_ROOT/(f"{config.data_root.stem}_"+ post_fix)
if not config.save_fold.exists():
    config.save_fold.mkdir(parents=True, exist_ok=True)


# In[6]:


fmodel = load_model(is_Ysq=False);
gmodel = load_model(is_Ysq=True);

dataset = get_dataset(
    data_path=config.data_root, 
    mask_type=config.mask_params.mask_type, 
    center_fraction=config.mask_params.center_fraction, 
    acceleration=config.mask_params.acceleration)

print(f"There is {len(dataset)} samples in the dataset.")
pprint(config)

# In[7]:


# save the config file 
save_pkl(config.save_fold/"config.pkl", config);


# In[12]:


def _run_fn(batch): 
    fn = batch.fname.split(".")[0]
    sn = batch.slice_num
    fn_root = config.save_fold/f"{fn}-{sn}.pkl"

    if fn_root.exists():
        print(f"{fn_root} exists, skip it.")
        return

    fields = batch._fields
    res = edict()
    res.fx = run_varnet_model(batch, fmodel)
    res.gx = run_varnet_model(batch, gmodel, is_Ysq=True);
    res.target = batch.target.numpy();
    res.mask = batch.mask.numpy();

    res.attrs = edict()
    for fv in fields:
        # I do not save masked_kspace (x), as it is very large
        if fv in ["mask", "target", "masked_kspace"]:
            continue
        v = getattr(batch, fv)
        if isinstance(v, torch.Tensor):
            v = v.numpy()
        res.attrs[fv] = v

    save_pkl(fn_root, res, is_force=False)
    return None


# In[13]:


n_data = len(dataset)
n_jobs = args.n_jobs

Parallel(n_jobs=n_jobs)(delayed(_run_fn)(dataset[idx]) for idx in tqdm(range(n_data), total=n_data));
