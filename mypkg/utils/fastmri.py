# this file includes functions to run inference on a pretrained model based on the fastMRI dataset


from constants import DATA_ROOT
from my_pl_modules.varnet_module_Ysq import VarNetModuleYsq

import torch
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data import SliceDataset
import fastmri.data.transforms as T
from fastmri.models import VarNet

def get_dataset(data_path, 
                mask_type="equispaced", 
                center_fraction=0.04, 
                acceleration=4):
    """Get the dataset from the given file path, mask type, center fractions and accelerations.
    Note that the input should be compatible with model training.
    - args: 
        - data_path (str): the path to the data
            - path to h5 files directory
        - mask_type (str): the type of mask to use, "equispaced" or "random"
            - we use "equispaced" for training the brain data
        - center_fraction (int): the center fraction to use
            - we use [0.04, 0.08] for training the brain data
        - acceleration (float): the acceleration to use
            - we use [4, 8] for training the brain data
    - returns: 
        - dataset (SliceDataset): the dataset to use for training
            it is a SliceDataset object, you can get data by calling dataset[i]
    """
    assert mask_type in ["equispaced", "random"], "mask_type should be either 'equispaced' or 'random'"
    assert center_fraction in [0.04, 0.08], "center_fraction should be either 0.04 or 0.08"
    assert acceleration in [4, 8], "acceleration should be either 4 or 8"

    mask = create_mask_for_mask_type(
            mask_type_str=mask_type, 
            center_fractions = [center_fraction], 
            accelerations = [acceleration]
        )
    data_transform = T.VarNetDataTransform(mask_func=mask)
    dataset = SliceDataset(
            root=data_path, 
            transform=data_transform, 
            challenge="multicoil", 
            sample_rate=None, 
            raw_sample_filter=None
        )
    return dataset


def run_varnet_model(batch, 
                     model, 
                     is_Ysq = False):
    """
    get the output of the model
    - args: 
        - batch: the batch of data
            get it from data loader
        - model: the trianed model
        - is_Ysq: if the model trained for Y^2 or not 
            - if True, the model will output Y^2
    - return:
        - output: the output of the model
    """
    mask = batch.mask
    masked_kspace = batch.masked_kspace
    crop_size = batch.crop_size
    if batch.mask.dim() == 4:
        mask = mask[None]
        masked_kspace = masked_kspace[None]
    else:
        assert batch.mask.shape[0] == 1, "currently only support batch size 1"
    model.eval()
    with torch.no_grad():
        output = model(masked_kspace, mask).cpu()
    # detect FLAIR 203
    if output.shape[-1] < crop_size[1]:
        crop_size = (output.shape[-1], output.shape[-1])

    output = T.center_crop(output, crop_size)[0]
    if is_Ysq:
        output = output**2
    return output.numpy()


def load_model(is_Ysq=True):
    """Load trained model 
    - args: 
        - is_Ysq (bool): whether to load the model trained with Ysq loss or not 
            - if True, load the model trained with Ysq loss, i.e., EY^2|X
            - if False, load the model trained with Y loss, i.e., EY|X
    - return:
        - model (VarNet): the trained model
    """
    # initial the model
    
    if is_Ysq: 
        model_raw = VarNetModuleYsq.load_from_checkpoint(DATA_ROOT/"pretrained_model/gmodel.ckpt");
        model = model_raw.varnet
    else:
        model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8);
        if torch.cuda.is_available():
            params = torch.load(DATA_ROOT/"pretrained_model/fmodel.pt")
        else:
            params = torch.load(DATA_ROOT/"pretrained_model/fmodel.pt", 
                                map_location=torch.device('cpu'))
        model.load_state_dict(params)
    model.eval()
    return model