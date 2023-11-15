from helpers import plotframes, plotframes_tensorboard, calculate_mean_std
from mvt import MaskedVideoTransformer
from gameds import GameDS
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
import argparse
from dual_logger import DualLogger
import matplotlib.pyplot as plt
import uuid
from torchvision import transforms

parser = argparse.ArgumentParser(prog='RunMVT', description='Trains the MVT model')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_frames', type=int, default=2)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--ddim', type=int, default=736)

parser.add_argument('--enc_heads', type=int, default=12)
parser.add_argument('--enc_layers', type=int, default=12)
parser.add_argument('--enc_ff', type=int, default=2048)

parser.add_argument('--dec_heads', type=int, default=12)
parser.add_argument('--dec_layers', type=int, default=12)
parser.add_argument('--dec_ff', type=int, default=2048)

parser.add_argument('--tensorboard', type=bool, default=True)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--run_name', type=str, default=None)

args = parser.parse_args()

run_name = f'./logs/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_{str(uuid.uuid4())}'
if args.run_name is not None:
    run_name = './logs/' + args.run_name + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

writer = DualLogger(log_dir=run_name, project_name='my_project')
if not args.tensorboard:
    writer = DualLogger(log_dir=f'/tmp/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_{str(uuid.uuid4())}', project_name='my_project')

args_str = "\n".join(f"{k}: {v}" for k, v in vars(args).items())
writer.tensorboard_writer.add_text('Hype[0.229, 0.224, 0.225]rparameters', args_str, 0)



from torch.utils.data import DataLoader

model = MaskedVideoTransformer(NUM_FRAMES=args.n_frames, COLOR_CHANNELS=1, D_DIM=args.ddim, PATCH_SIZE=args.patch_size,
            ENC_HEADS=args.enc_heads, ENC_LAYERS=args.enc_layers, ENC_FF=args.enc_ff, DEC_HEADS=args.dec_heads, DEC_LAYERS=args.dec_layers, DEC_FF=args.dec_ff)
model.to(args.device);

import wandb
from torch.utils.tensorboard import SummaryWriter

print("run started")



from PIL import Image
from helads import HelaData


train_dir = "/home/constantin/Documents/celltracking/HeLa_dataset/train"
test_dir = "/home/constantin/Documents/celltracking/HeLa_dataset/test"

#train_dir = "/scratch1/projects/cca/data/tracking/microscopy/Sartorius-DFKI/Tracking_datasets/HeLa_dataset/train"
#test_dir = "/scratch1/projects/cca/data/tracking/microscopy/Sartorius-DFKI/Tracking_datasets/HeLa_dataset/test"


hela_train = HelaData(train_dir, sequence_length=args.n_frames)
hela_val = HelaData(test_dir, sequence_length=args.n_frames)
train_dataloader = DataLoader(hela_train, batch_size=24, shuffle=True, prefetch_factor=None, num_workers=0)
val_dataloader = DataLoader(hela_val, batch_size=24, shuffle=True, prefetch_factor=None, num_workers=0)

if args.normalize:
    """ get mean and std """
    hela_tmp = HelaData(train_dir, sequence_length=1)
    tmp_dataloader = DataLoader(hela_tmp, batch_size=24, shuffle=True, prefetch_factor=None, num_workers=0)
    mean, std = calculate_mean_std(tmp_dataloader)
    normalizer = transforms.Normalize(mean=mean, std=std)

print("Dataloader length", len(train_dataloader))

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)

step = 0
optim.zero_grad()
for _ in range(1000000):
    torch.save(model.state_dict(), "model.pt")
    for i_step, (X, _) in enumerate(train_dataloader):

        X = X.to(args.device)
        
        if args.normalize:
            X = normalizer(X)
        
        
        X_pred, (X_masked, ) = model(X)
        
        print("SHAPES", X_pred.shape, X_masked.shape)

        loss = criterion(X_pred, X)
        #print(loss.item())
    
        loss.backward()
        if step%20 == 19 or True:
            optim.step()
            optim.zero_grad()
 
        writer.add_scalar('Training loss', loss.item(), step)
    
        if step%20 == 10:
            image_tensor = plotframes_tensorboard(X_masked, f"Train: Mask {step}")
            writer.add_image('Train: Mask', image_tensor, step)

            image_tensor = plotframes_tensorboard(X_pred, f"Train: Prediction {step}")
            writer.add_image('Train: Prediction', image_tensor, step)

            image_tensor = plotframes_tensorboard(X, f"Train: Original {step}")
            writer.add_image('Train: Original', image_tensor, step)

            image_tensor = plotframes_tensorboard([X, X_pred, X_masked], f"Train: Combined {step}")
            writer.add_image('Train: Combined', image_tensor, step)
    
            # evaluation dataset
            X_eval, _ = random.choice(hela_val)
            if args.normalize:
       	        X_eval = normalizer(X_eval)

            X_eval = X_eval[None].to(args.device)
            X_eval_pred, (X_eval_pred_masked, ) = model(X_eval)

            
            print("before eval", X_eval.shape, X_eval_pred_masked.shape, X_eval_pred.shape)
            loss = criterion(X_eval_pred, X_eval[0])
            writer.add_scalar('Validation loss', loss.item(), step)
    
            image_tensor = plotframes_tensorboard(X_eval_pred_masked, f"Val: Mask {step}")
            writer.add_image('Val: Mask', image_tensor, step)
            
            image_tensor = plotframes_tensorboard(X_eval_pred, f"Val: Prediction {step}")
            writer.add_image('Val: Prediction', image_tensor, step)
        
            image_tensor = plotframes_tensorboard(X_eval, f"Val: Original {step}")
            writer.add_image('Val: Original', image_tensor, step)

            image_tensor = plotframes_tensorboard([X_eval, X_eval_pred, X_eval_pred_masked], f"Val: Combined {step}")
            writer.add_image('Eval: Combined', image_tensor, step)

            
            
            del X_eval_pred
            del X_eval_pred_masked

        torch.save(X_masked, 'tensor.pt')

            

        del X_pred
        #del X_masked
        step += 1
    

