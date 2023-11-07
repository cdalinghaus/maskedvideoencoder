import wandb
from torch.utils.tensorboard import SummaryWriter

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
