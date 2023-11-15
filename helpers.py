import matplotlib.pyplot as plt
import torch
import io
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0

    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
 
    mean /= total_images_count
    std /= total_images_count

    return mean, std


def plotframes(X, title=None, show=True):
    
    if not (type(X) == list):
        X = [X]
    
    batch_size, frames, color_channels, x, y = X[0].shape

    fig, ax = plt.subplots(len(X), frames, figsize=(frames * 3, len(X)*3))

    for imaxis, x in enumerate(X):
        for idx in range(frames):
            if len(X) > 1:
                ax[imaxis, idx].imshow(x[0, idx].permute(2,1,0).cpu().detach().numpy())
                ax[imaxis, idx].get_xaxis().set_visible(False)
                ax[imaxis, idx].get_yaxis().set_visible(False)
            else:
                ax[idx].imshow(x[0, idx].permute(2,1,0).cpu().detach().numpy())
                ax[idx].get_xaxis().set_visible(False)
                ax[idx].get_yaxis().set_visible(False)
    
    plt.subplots_adjust(top = 0.92, hspace=0.05, wspace=0.02)
    
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

def patchify(X, patch_size=32):
    X = X.moveaxis(2,1)
    
    #batch_size, frame, color_channel, width, height = X.shape
    batch_size, color_channel, frame, width, height = X.shape
    
    patches_per_dim = width // patch_size
    
    # Unfold over spacial dimension
    patches = X.unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    
    # Combine the axis
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6)
    patches = patches.reshape(batch_size, frame * patches_per_dim * patches_per_dim, color_channel, patch_size, patch_size)
    
    return patches

def unpatchify(X, img_size=128, patch_size=32, num_frames=2, color_channels=1):
    batch_size, num_patches, color_channel, patch_width, patch_height = X.shape
    patches_per_dim = img_size // patch_width
    
    # Change shape
    patches = X.reshape(batch_size, num_frames, patches_per_dim, patches_per_dim, color_channel, patch_size, patch_size)
    patches = patches.permute(0, 4, 1, 2, 3, 5, 6)
    
    # Concatenate the patches along their original axis
    intermediate = torch.concat([patches[:, :, :, i, :, :, :] for i in range(patches_per_dim)], axis=4)
    result = torch.concat([intermediate[:, :, :, i, :, :] for i in range(patches_per_dim)], axis=4)
    
    return result.moveaxis(2,1)
