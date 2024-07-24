from nerf.plenoxels import Plenoxels 
from nerf.hash import Hash
from nerf.nglod import NGLOD
from nerf.mlp_net import TinyMLP 
import torch



def create_model(model_type , params):

    base_lod = params.get('base_lod')
    num_lod = params.get('num_lod')
    feature_dim = params.get('feature_dim')

    near = params.get('near')
    far = params.get('far')
    nb_epochs = params.get('nb_epochs')
    lr = params.get('lr')
    gamma = params.get('gamma')


    
    if model_type == "mlp":
        return TinyMLP()
    elif model_type == "plenoxels":
        

        model = Plenoxels([100] , 4, 3 , feature_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    elif model_type == "nglod":

        model = NGLOD (base_lod , num_lod , feature_dim , L = 4 , scene_scale = 3)
        optimizer = torch.optim.Adam(
    [{"params": model.codebook.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 0.},
      {"params": model.sigma_mlp.parameters(), "lr": 1e-2,  "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6},
      {"params": model.pred_color_mlp.parameters(), "lr": 1e-2,  "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6}])

    elif model_type == "hash":
        return Hash()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)

    return model , optimizer , scheduler

if __name__ == "__main__":
    model_type = input("Enter model type (mlp, plenoxels, nglod, hash): ").strip().lower()
    try:
        model = create_model(model_type)
        print(f"{model_type.upper()} model created successfully.")
    except ValueError as e:
        print(e)
