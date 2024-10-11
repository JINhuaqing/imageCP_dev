from math import fmod
import sys
from jin_utils import get_mypkg_path
pkgpath = get_mypkg_path()
sys.path.append(pkgpath)
from constants import RES_ROOT, DATA_ROOT, FIG_ROOT


import time
from collections import defaultdict
from easydict import EasyDict as edict
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict as ddict
import pandas as pd

from utils.misc import save_pkl, load_pkl, num2str, str2num
plt.style.use(FIG_ROOT/"base.mplstyle")

def _run_fn(res_fold):
    def _sort_fn(fil):
        parts = fil.stem.split("_")
        nums = []
        for p in parts: 
            index = p.find("-")
            nums.append(str2num(p[(index+1):]))
        return nums
    fils = list(res_fold.glob("size*.pkl"));
    fils = sorted(fils, key=_sort_fn);


    # extract the data into a dataframe
    df_dict = ddict(list)
    for fil in fils: 
        res = load_pkl(fil, verbose=False)
        for re in res: 
            for ky in ["cp-spl", "cpsemi-nospl", "cp-nospl", "cpsemi-spl"]:
                if ky not in re.keys(): continue
                df_dict["eps"].append(re[ky]["eps"])
                df_dict["empcv"].append(re[ky]["in_sets"])
                df_dict["method"].append(ky)
                if "size" in re["info"]:
                    df_dict["size"].append(re["info"]["size"])
                else:
                    df_dict["size"].append(re["info"]["ntrain"])
                df_dict["hfct"].append(re["info"]["hfct"])
                df_dict["seed"].append(re["info"]["seed"])
                #df_dict["h"].append(re["info"]["h"])

    df = pd.DataFrame(df_dict);
    hfcts = np.sort(np.unique(df["hfct"]))
    sizes = np.sort(np.unique(df["size"]))


    def _sort_fn(x, is_print=False):
        """Sort the method names
        """
        if x.startswith("cp-nospl"):
            v = "cp-nospl" if is_print else -1
        elif x.startswith("cp-spl"): 
            v = "cp-spl" if is_print else -2
        elif x.startswith("cpsemi"):
            hfct = float(x.split("-")[-1])
            v = f"{hfct:.3E}" if is_print else hfct
        return v
    # let us combine h and method 
    df["method1"] = df.apply(lambda x: x["method"]+"-"+str(x["hfct"]), axis=1)

    # let remve some redundant data for naive method, that is the results from all hfcts are the same for naive method
    # so we only need to keep the results from the first hfct
    kpidxs = np.bitwise_and(df["hfct"]!=hfcts[0], df["method"].apply(lambda x: "cpsemi" not in x))
    kpidxs = np.bitwise_not(kpidxs)
    kpidxs = np.bitwise_and(kpidxs, df["method"]!="cpsemi-nospl")
    kpidxs = np.bitwise_and(kpidxs, df["method"]!="cp-nospl")

    for size in sizes:
        kpidxs0 = np.bitwise_and(kpidxs, df["size"]==size)
        sel_df = df[kpidxs0]

        names = np.unique(sel_df["method1"])
        order_names = sorted(names, key=_sort_fn)
        print_names = [_sort_fn(name, is_print=True) for name in order_names]

        vars = sel_df.groupby(["method1"])["eps"].var()
        vars = np.array(vars[order_names])
        rl_vars = vars[0]/vars

        cvs = sel_df.groupby(["method1"])["empcv"].mean()
        cvs = np.array(cvs[order_names])
        # add cv in print_names
        print_names = [f"{print_names[i]} ({cvs[i]:.2f})" for i in range(len(print_names))]

        fig, ax1 = plt.subplots()
        if np.max(rl_vars) > 1.01: 
            plt.title(f"Eps vs Method when size={size} (max rel.var. is {np.max(rl_vars):.2f})")
        else:
            plt.title(f"Eps vs Method when size={size}")
        sns.boxplot(data=sel_df, x="method1", y="eps", order=order_names, showfliers=False, ax=ax1)
        ax1.set_xticks(ticks=range(len(order_names)), labels=print_names, rotation=45);
        ax1.set_ylabel("EPS")
        ax1.set_xlabel("Method")

        ax2 = ax1.twinx()
        ax2.plot(rl_vars, "ro-")
        ax2.axhline(1, color='r', linestyle='--')
        ax2.set_ylabel("Rel. Var. (w.r.t. CP-spl)", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        fig.tight_layout()  
        plt.savefig(FIG_ROOT/f"{res_fold.stem}_size{size}.jpg")
        #plt.savefig(res_fold/f"eps_vs_method_size{size}.jpg")
        plt.close()

X_types = ["normal", "binary"]
kernels = ["None", "diff"]
noise_types = ["normal", "lognormal"]
fmodel_types = ["LIN", "MLP"]


from itertools import product
all_items = product(X_types, kernels, noise_types, fmodel_types)
for items in all_items:
    X_type, kernel, noise_type, fmodel_type = items
    print(items)
    res_fold = RES_ROOT/f"ExpSqFixA_X{X_type}_sizexy-50x5_noise-0d5_kernel-{kernel}_fmodel-{fmodel_type}_noise-{noise_type}"
    _run_fn(res_fold)