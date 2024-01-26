import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def get_img_to_tensor_transform():
    return transforms.Compose([transforms.PILToTensor()])


def class_index_to_one_hot(index: int, classes_count: int) -> []:
    """
    Return one hot encoding
    """

    if index >= classes_count:
        index = classes_count-1

    targets = np.array([index]).reshape(-1)
    one_hot_targets = np.eye(classes_count)[targets]

    return one_hot_targets[0]

def get_2D_image_of_last_3D_img_in_batch(batch: torch.Tensor, image_index: int = -1, squeeze=True):
    transform = transforms.ToPILImage()
    if squeeze:
        last_image = torch.squeeze(batch.cpu().detach())[-1, :, :, :, image_index]
    else:
        last_image = batch.cpu().detach()[-1, :, :, :, image_index]
    img = transform(last_image)
    return img

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst