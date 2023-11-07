import os
import torch
from PIL import Image
import numpy as np

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

class HelaData:

    def __init__(self, base_dir="train", sequence_length=10):
        self.sequence_length = sequence_length
        self.sequences = []
        self.sequence_start_indices = []
        self.current_start_index = 0
        self.base_dir = base_dir

        for burst in os.listdir(base_dir):
            burst_path = os.path.join(base_dir, burst, "img1")
            frames = sorted(os.listdir(burst_path))
            frames = [os.path.join(base_dir, burst, "img1", x) for x in frames]
            self.sequences.append(frames)
            self.sequence_start_indices.append(self.current_start_index)
            self.current_start_index += len(frames) - self.sequence_length
        #print(self.sequence_start_indices)
        
    def __getitem__(self, idx):
        #print(max([x for x in self.sequence_start_indices if x <= idx]))
        start_item = max([x for x in self.sequence_start_indices if x <= idx])
        start_index = self.sequence_start_indices.index(start_item)
        
        local_index = idx - start_item
        files = self.sequences[start_index][local_index:local_index+self.sequence_length]
        
        images = [resize_to_canvas(Image.open(f)) for f in files]
        tensors = [torch.tensor(np.array(im)) for im in images]
        stacked = torch.stack(tensors)
        result = stacked
        result = result.moveaxis(3, 1)[:, :, :336, :464]
        return result, torch.Tensor([0])
        
    def __len__(self):
        return self.current_start_index
    
ds = HelaData("/home/constantin/Documents/celltracking/HeLa_dataset/train")

ds[0]
