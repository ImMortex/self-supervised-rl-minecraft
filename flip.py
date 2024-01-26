import torch
from torchvision import transforms
from PIL import Image


# Load your image (replace 'path_to_your_image.jpg' with the actual path)
image_path = './tmp/tis_inventory_open.png'
img = Image.open(image_path)

# Convert PIL Image to PyTorch Tensor
img_tensor = transforms.ToTensor()(img)

# Flip the image horizontally at 10 pixels
flip_point_h = -20

if flip_point_h > 0:
    img_tensor = torch.cat([img_tensor[:, :, flip_point_h:], img_tensor[:, :, -flip_point_h:].flip(2)], dim=2)
if flip_point_h < 0:
    img_tensor = torch.cat([img_tensor[:, :, :-flip_point_h].flip(2), img_tensor[:, :, :flip_point_h]], dim=2)
flip_point_v = 0
if flip_point_v > 0:
    img_tensor = torch.cat([img_tensor[:, flip_point_v:, :], img_tensor[:, -flip_point_v:, :].flip(1)], dim=1)
if flip_point_v < 0:
    img_tensor = torch.cat([img_tensor[:, :-flip_point_v, :].flip(1), img_tensor[:, :flip_point_v, :]], dim=1)

# Convert PyTorch Tensor back to PIL Image
flipped_img = transforms.ToPILImage()(img_tensor )

flipped_img.show()