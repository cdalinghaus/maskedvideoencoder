import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

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

class FixedTransform():
    def __init__(self, min_angle, max_angle, crop_height, crop_width):
        # Generate fixed parameters for all transformations
        self.angle = torch.FloatTensor(1).uniform_(min_angle, max_angle).item()
        self.position = None
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, img):
        # If position hasn't been set, choose a random position
        if self.position is None:
            w, h = img.size
            th, tw = self.crop_height, self.crop_width
            if w == tw and h == th:
                self.position = (0, 0)
            else:
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                self.position = (i, j)

        # Apply the same transformation to the image
        transformed_img = transforms.functional.rotate(img, self.angle)
        transformed_img = transforms.functional.crop(transformed_img, *self.position, self.crop_height, self.crop_width)
        transformed_img = transforms.ToTensor()(transformed_img)

        return transformed_img

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
        
    def _transform(self, img):
        # Define the transformations with padding and center cropping
        

        # Apply the transformation to the image
        transformed_image = transform(img)
        return transformed_image
        
    def __getitem__(self, idx):
        #print(max([x for x in self.sequence_start_indices if x <= idx]))
        start_item = max([x for x in self.sequence_start_indices if x <= idx])
        start_index = self.sequence_start_indices.index(start_item)
        
        local_index = idx - start_item
        files = self.sequences[start_index][local_index:local_index+self.sequence_length]
        
        images = [Image.open(f) for f in files]
        
        fixed_transformation = FixedTransform(min_angle=0, max_angle=359, crop_height=128, crop_width=128)
        #print(fixed_transformation)
        
        #tensors = [fixed_transformation(im) for im in images]
        tensors = []
        for file_path in files:
            with Image.open(file_path) as img:  # This will ensure the file is closed after the block
                tensor = fixed_transformation(img)
                tensors.append(tensor)

        stacked = torch.stack(tensors)
        #result = stacked
        #result = result.moveaxis(3, 1)
        return stacked, torch.Tensor([0])
        
    def __len__(self):
        return self.current_start_index

