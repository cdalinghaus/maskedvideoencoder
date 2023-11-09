import matplotlib.pyplot as plt
import torch
import io
from PIL import Image
import torchvision.transforms as transforms

def plotframes(X, title=None, show=True):
    batch_size, frames, color_channels, x, y = X.shape
    
    fig, ax = plt.subplots(1, frames, figsize=(frames * 3, 3))
    
    for idx in range(frames):
        if frames > 1:
            ax.flat[idx].imshow(X[0, idx].permute(2,1,0).cpu().detach().numpy())
        else:
            ax.imshow(X[0, idx].permute(2,1,0).cpu().detach().numpy())
    if title is not None:
        fig.suptitle(title)
    if show:
        plt.show()

def plotframes_tensorboard(X, title=None):
    # Create a buffer to save the figure to
    buffer = io.BytesIO()

    plotframes(X, title, show=False)

    # Save the current figure to the buffer
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Move the buffer's position to the beginning
    buffer.seek(0)

    # Load the buffer content as an image
    image = Image.open(buffer)
    
    # Convert the image to a PyTorch tensor
    tensor = transforms.ToTensor()(image)

    # Close the buffer and figure
    buffer.close()
    plt.close()

    return tensor

def reconstruct_from_patches(patches, IMG_SIZE, PATCH_SIZE, NUM_FRAMES, COLOR_CHANNELS):
    """
    Reconstruct the video tensor from its patches.
    
    Parameters:
    - patches: the patches tensor.
    - IMG_SIZE: the original dimension of the image (height or width, assuming they are the same).
    - PATCH_SIZE: the size of each patch.
    
    Returns:
    - video_tensor: the reconstructed video tensor.
    """
    
    patches_per_dim = IMG_SIZE // PATCH_SIZE
    batch_size = patches.shape[0]
    
    # Reshape patches to prepare for 'folding'
    patches_reshaped = patches.reshape(batch_size, NUM_FRAMES, patches_per_dim, patches_per_dim, COLOR_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    
    # Fold the patches back into full frames
    video_tensor = patches_reshaped.permute(0, 1, 4, 2, 5, 3, 6).reshape(batch_size, NUM_FRAMES, COLOR_CHANNELS, IMG_SIZE, IMG_SIZE)
    
    return video_tensor
