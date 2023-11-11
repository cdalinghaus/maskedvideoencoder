from helpers import plotframes, reconstruct_from_patches, plotframes_tensorboard
from mvt import MaskedVideoTransformer
from gameds import GameDS
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
import argparse
from dual_logger import DualLogger

parser = argparse.ArgumentParser(prog='RunMVT', description='Trains the MVT model')
parser.add_argument('--device', type=str, default="cuda")

args = parser.parse_args()



from torch.utils.data import DataLoader

model = MaskedVideoTransformer(NUM_FRAMES=1, COLOR_CHANNELS=1)
model.to(args.device);

import wandb
from torch.utils.tensorboard import SummaryWriter

print("run started")

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
train_dataloader = DataLoader(hela_train, batch_size=240, shuffle=True, prefetch_factor=None, num_workers=0)
val_dataloader = DataLoader(hela_val, batch_size=240, shuffle=True, prefetch_factor=None, num_workers=0)

print(len(train_dataloader))

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(params=model.parameters(), lr=0.01)

step = 0
optim.zero_grad()
for _ in range(1000000):
    torch.save(model.state_dict(), "model.pt")
    for i_step, (X, _) in enumerate(train_dataloader):
        print(i_step)
        X = X.to(args.device)
        
        
        X_pred, (X_masked, ) = model(X)
        
        loss = criterion(X_pred, X)
    
        loss.backward()
        if step%20 == 19:
            optim.step()
            optim.zero_grad()
            print(step)
 
        writer.add_scalar('Training loss', loss.item(), step)
    
        if step%20 == 18:
            image_tensor = plotframes_tensorboard(X_masked, f"Train: Mask {step}")
            writer.add_image('Train: Mask', image_tensor, step)
            
            image_tensor = plotframes_tensorboard(X_pred, f"Train: Prediction {step}")
            writer.add_image('Train: Prediction', image_tensor, step)
        
            image_tensor = plotframes_tensorboard(X, f"Train: Original {step}")
            writer.add_image('Train: Original', image_tensor, step)
    
            # evaluation dataset
            X_eval, _ = random.choice(hela_val)
            X_eval = X_eval[None].to(args.device)
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
    

