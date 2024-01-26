import numpy as np

def mse_imgs(image_1: np.ndarray, image_2: np.ndarray) -> float:
    """
    Calculates the mean squared error between two images
    @param image_1: np.ndarray
    @param image_2: np.ndarray
    @return: float
    """
    error: float = np.sum((image_1.astype("float") - image_2.astype("float")) ** 2)
    error /= float(image_1.shape[0] * image_1.shape[1])
    return error
