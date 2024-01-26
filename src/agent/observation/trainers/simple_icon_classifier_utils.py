import cv2
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


def custom_preprocess_data_x(images: [], config_dict=None):
    """
    Resizes image and black out minecraft item amount in lower rigth corner

    @param images:
    @return:
    """

    if config_dict is None:
        config_dict = {}
    results: [] = []
    for img in images:
        result = resize_img(config_dict, img, config_dict["width"], config_dict["height"])

        if "black_out_item_amount" not in config_dict or config_dict["black_out_item_amount"]:
            start_point = (
                int(config_dict["width"] / 4), int(config_dict["height"] / 2))  # top left corner of rectangle
            end_point = (config_dict["width"], config_dict["height"])  # bottom right corner of rectangle
            color = (0, 0, 0)  # Black color in BGR
            thickness = -1  # Thickness of -1 will fill the entire shape
            result = cv2.rectangle(result, start_point, end_point, color, thickness)
        # cv2.imshow("window", result)
        # cv2.waitKey(-1)
        result = cv2.normalize(result, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # Normalization
        results.append(result)

    return results


def custom_augment_data_x(images: [], resize_width=None, resize_height=None, config_dict={}):
    """
    Resizes image and resize it back to input shape change the picture a little

    @param images:
    @return:
    """

    results: [] = []
    for img in images:
        if resize_width is not None and resize_height is not None:
            # resize larger
            img = resize_img(config_dict, img, resize_width, resize_height)
            # resizer smaller
            img = resize_img(config_dict, img, config_dict["width"], config_dict["height"])

        if config_dict["channels"] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results.append(img)
    return results


def cv_img_list_to_torch_tensor(images: []) -> torch.Tensor:
    tensor = torch.Tensor(np.array(images))

    if len(tensor.shape) == 3:  # if grayscale
        tensor = tensor[:, :, :, None]  # add missing channel dim
    return tensor.permute(0, 3, 1, 2)


def predict(int_2_label, model, test_x):
    predicted_labels = []
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode

        # loop over the test set. For each batch
        for x in test_x:
            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            predictions_tensor = model(x)
            predictions = predictions_tensor.detach().cpu().numpy()
            predicted_labels += one_hots_to_labels(int_2_label, predictions)
    return predicted_labels


def resize_data_x(images: [], config_dict: dict):
    """
    Resizes images
    """

    if config_dict is None:
        config_dict = {}

    results: [] = []
    for img in images:
        result = resize_img(config_dict, img, config_dict["width"], config_dict["height"])
        results.append(result)

    return results


def resize_img(config_dict, img, width, height):
    if "resize_interpolation" in config_dict and config_dict["resize_interpolation"] == "INTER_LANCZOS4":
        result = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    else:
        result = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return result


def get_model(classes_count, model_name, in_channels=3) -> nn.Module:
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    pytorch_vision: str = 'pytorch/vision:v0.10.0'
    # added additional linear layer because classes_count > standard resnet out classes 1000
    if model_name == 'resnet18':
        model = torch.hub.load(pytorch_vision, 'resnet18', pretrained=False)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # channels
        model = nn.Sequential(model, nn.Linear(1000, classes_count))  # modify res net output labels
        return model

    if model_name == 'resnet50':
        model = torch.hub.load(pytorch_vision, 'resnet50', pretrained=False)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # channels
        model = nn.Sequential(model, nn.Linear(1000, classes_count))  # modify res net output labels
        return model

    if model_name == 'resnet101':
        model = torch.hub.load(pytorch_vision, 'resnet101', pretrained=False)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # channels
        model = nn.Sequential(model, nn.Linear(1000, classes_count))  # modify res net output labels
        return model

    return None


def one_hots_to_labels(int_2_label, predictions):
    predicted_labels = []
    for one_hot_encoding_array in predictions:
        label = one_hot_to_label(int_2_label, one_hot_encoding_array)
        predicted_labels.append(label)
    return predicted_labels


def one_hot_to_label(int_2_label, one_hot_encoding_array):
    index = one_hot_encoding_array.argmax(axis=None)
    label = int_2_label[str(index)]
    return label
