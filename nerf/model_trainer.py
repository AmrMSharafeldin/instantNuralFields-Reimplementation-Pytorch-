import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
from einops import rearrange, reduce, repeat
from tqdm import tqdm
from nerf.volume_renderer import volume_renderer

from nerf.eval import test
from utils import * 

class NerfTrainRunner:
    def __init__(self, model, optimizer, scheduler, near, far, nb_bins, max_epoches, data_loader, test_o,test_d ,test_target , batch_size ,checkpoints_path, loss_fn = mse ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.near = near
        self.far = far
        self.nb_bins = nb_bins
        self.max_epoches = max_epoches
        self.data_loader = data_loader
        self.checkpoints_path = checkpoints_path
        self.training_loss = []
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.test_o = test_o
        self.test_d = test_d
        self.test_target = test_target
        self.psnrs = []
        self.iternums = []


    def training_step(self, batch):
        o = batch[:, :3].to(self.device)
        d = batch[:, 3:6].to(self.device)
        target = batch[:, 6:].to(self.device)
        prediction = volume_renderer(self.model, o, d, self.near, self.far, bins=self.nb_bins)
        loss = mse(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_loss.append(loss.item())

    def train(self):

        num_batches = len(self.data_loader)
        interval = num_batches // 10  # Calculate the interval for 25%

        for epoch in range(self.max_epoches):
            for i, batch in enumerate(tqdm(self.data_loader), start=1):
                loss = self.training_step(batch)
                if i % interval == 0:
                  with torch.no_grad():
                    img, mse, psnr = test(self.model, torch.from_numpy(self.test_o).to(self.device).float(),
                    torch.from_numpy(self.test_d).to(self.device).float(), near = self.near , far = self.far ,target=self.test_target.reshape(400, 400, 3))
                    self.iternums.append(i)
                    self.psnrs.append(psnr)
                    self.plot_training_progress(img, i )
                #self.save_checkpoint(i)
            self.scheduler.step()

            # Render the holdout view for logging



            # End of logging section

        return self.training_loss


    def save_checkpoint(self, epoch):
        # Save the entire model
        checkpoint_path = os.path.join(self.checkpoints_path, f'model_nerf_epoch_{epoch}.pth')
        torch.save(self.model.cpu(), checkpoint_path)
        self.model.to(self.device)

        # Save the model state dictionary and other relevant information
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
            },
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth")
        )

    def plot_loss(self):
            plt.figure(figsize=(10, 5))
            plt.plot(self.training_loss, label='Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training Loss over Time')
            plt.legend()
            plt.show()

    def plot_training_progress( self , rgb, iteration):
      """
      Plots the training progress showing the RGB image and PSNR values.

      Args:
          rgb (np.ndarray): The RGB image to display.
          iteration (int): The current iteration number.
          iternums (list[int]): List of iteration numbers for PSNR plotting.
          psnrs (list[float]): List of PSNR values corresponding to the iterations.
      """
      plt.figure(figsize=(10, 4))

      # Plot the RGB image
      plt.subplot(121)
      plt.imshow(rgb)
      plt.title(f'Iteration: {iteration}')

      # Plot the PSNR values
      plt.subplot(122)
      plt.plot(self.iternums, self.psnrs)
      plt.title('PSNR')
      plt.xlabel('Iteration')
      plt.ylabel('PSNR')

      plt.show()


    def run(self):
      img, mse, psnr = test(self.model, torch.from_numpy(self.test_o).to(self.device).float(),
      torch.from_numpy(self.test_d).to(self.device).float(), near = self.near , far = self.far ,target=self.test_target.reshape(400, 400, 3))
      self.iternums.append(0)
      self.psnrs.append(psnr)
      self.plot_training_progress(img, 0 )
      print("test")
      self.train()
      self.plot_loss()
