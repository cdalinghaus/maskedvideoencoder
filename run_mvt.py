from helpers import plotframes, reconstruct_from_patches, plotframes_tensorboard
from mvt import MaskedVideoTransformer
from gameds import GameDS
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random

device = "cuda"
device = "cpu"

from torch.utils.data import DataLoader

model = MaskedVideoTransformer(NUM_FRAMES=1)
model.to(device);

import wandb
from torch.utils.tensorboard import SummaryWriter

print("run started")
class DualLogger:
    def __init__(self, log_dir=None, project_name=None):
        """
        log_dir: Directory for TensorBoard logs
        project_name: Weights & Biases project name
        """
        self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
        
        if project_name:
            wandb.init(project=project_name)
    
    def add_scalar(self, tag, scalar_value, global_step=None):
        # Logging to TensorBoard
        self.tensorboard_writer.add_scalar(tag, scalar_value, global_step)

        # Logging to wandb
        wandb.log({tag: scalar_value}, step=global_step)
    
    def add_image(self, tag, img_tensor, global_step=None):
        # Logging to TensorBoard
        self.tensorboard_writer.add_image(tag, img_tensor, global_step)

        # Logging to wandb. Note: wandb requires image data to be in PIL or numpy format.
        # Assuming img_tensor is a PyTorch tensor, we can convert it to a wandb-compatible format.
        image = wandb.Image(img_tensor.permute(1, 2, 0).cpu().numpy())  # Assuming img_tensor is of shape (C, H, W)
        wandb.log({tag: image}, step=global_step)
    
    def close(self):
        self.tensorboard_writer.close()
        wandb.finish()

# Usage:
writer = DualLogger(log_dir='./logs', project_name='my_project')
#writer.add_scalar('Training loss', 0.5, 1)
#writer.add_image('Train: Mask', image_tensor, 1)  # image_tensor should be a PyTorch tensor.
#writer.close()


import matplotlib.pyplot as plt

#i = iter(train_dataloader)

from PIL import Image
"""
def resize_to_canvas(image, canvas_size=(256, 256)):
    # Get the aspect ratio of the image
    aspect_ratio = image.width / image.height

    if image.width > image.height:  # Landscape or square
        new_width = canvas_size[0]
        new_height = int(canvas_size[0] / aspect_ratio)
    else:  # Portrait
        new_height = canvas_size[1]
        new_width = int(canvas_size[1] * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new blank canvas
    canvas = Image.new("RGB", canvas_size, "white")

    # Compute the position to paste the resized image onto the canvas
    x_offset = (canvas_size[0] - new_width) // 2
    y_offset = (canvas_size[1] - new_height) // 2

    canvas.paste(resized_image, (x_offset, y_offset))
    return canvas
"""
from helads import HelaData

hela_train = HelaData("/scratch1/projects/cca/data/tracking/microscopy/Sartorius-DFKI/Tracking_datasets/HeLa_dataset/train", sequence_length=1)
hela_val = HelaData("/scratch1/projects/cca/data/tracking/microscopy/Sartorius-DFKI/Tracking_datasets/HeLa_dataset/test", sequence_length=1)
train_dataloader = DataLoader(hela_train, batch_size=24, shuffle=True, prefetch_factor=4, num_workers=4)
val_dataloader = DataLoader(hela_val, batch_size=24, shuffle=True, prefetch_factor=4, num_workers=4)

print(len(train_dataloader))

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(params=model.parameters(), lr=0.01)

step = 0
for _ in range(1000):
    for i_step, (X, _) in enumerate(train_dataloader):
        print(i_step)
        X = X.to(device)
        
        
        X_pred, (X_masked, ) = model(X)
        
        loss = criterion(X_pred, X)
    
        loss.backward()
        if step%200 == 1:
            optim.step()
            optim.zero_grad()
            print(step)
 
        writer.add_scalar('Training loss', loss.item(), step)
    
        if step%100 == 1:
            image_tensor = plotframes_tensorboard(X_masked, f"Train: Mask {step}")
            writer.add_image('Train: Mask', image_tensor, step)
            
            image_tensor = plotframes_tensorboard(X_pred, f"Train: Prediction {step}")
            writer.add_image('Train: Prediction', image_tensor, step)
        
            image_tensor = plotframes_tensorboard(X, f"Train: Original {step}")
            writer.add_image('Train: Original', image_tensor, step)
    
            # evaluation dataset
            X_eval, _ = random.choice(hela_val)
            X_eval = X_eval / 255
            X_eval = X_eval[None].to(device)
            X_eval_pred, (X_eval_pred_masked, ) = model(X_eval)

            loss = criterion(X_eval_pred, X_eval)
            writer.add_scalar('Validation loss', loss.item(), step)
    
            image_tensor = plotframes_tensorboard(X_eval_pred_masked, f"Val: Mask {step}")
            writer.add_image('Val: Mask', image_tensor, step)
            
            image_tensor = plotframes_tensorboard(X_eval_pred, f"Val: Prediction {step}")
            writer.add_image('Val: Prediction', image_tensor, step)
        
            image_tensor = plotframes_tensorboard(X_eval, f"Val: Original {step}")
            writer.add_image('Val: Original', image_tensor, step)

            del X_eval_pred
            del X_eval_pred_masked


        del X_pred
        del X_masked
        step += 1
    

