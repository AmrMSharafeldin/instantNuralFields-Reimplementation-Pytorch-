from nerf.plenoxels import Plenoxels 
from nerf.hash import Hash
from nerf.nglod import NGLOD
from nerf.mlp_net import TinyMLP 
import torch



def create_model(model_type , params):

    base_lod = params.get('base_lod')
    num_lod = params.get('num_lod')

    gamma = params.get('gamma')
    lr = params.get('lr')


    
    if model_type == "mlp":
        return TinyMLP()
    elif model_type == "plenoxels":
        

        model = Plenoxels([100] , 4, 3 , 4)

    elif model_type == "nglod":
        model = NGLOD( base_lod , num_lod, L =4, scene_scale=3 , feature_dim = 4)

    elif model_type == "hash":
        return Hash()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
   
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)

    return model , optimizer , scheduler

