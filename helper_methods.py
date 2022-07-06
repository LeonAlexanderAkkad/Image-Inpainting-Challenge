import random

from typing import Tuple
import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_target_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def ex4(image_array: np.ndarray,
        offset: Tuple[int, int],
        spacing: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function processes and prepares images for training an CNN directly from these images.

    It creates two input arrays from a given input image. For this, it sets pixels that don't
    lie on a grid specified by the parameters offset and spacing to 0.

    Parameters
    ----------
    image_array: np.ndarray
        A 3D numpy array which holds the RGB image data.
    offset: Tuple[int, int]
        A tuple containing two int values which are used to specify the grid used in the function.
    spacing: Tuple[int, int]
        A tuple containing two int values which are used to specify the grid used in the function.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the input_array, known_array and image_array.

    Raises
    ------
    TypeError
        If image_array is not of type np.ndarray.
    NotImplementedError
        If there are issues with the shape.
    ValueError
        If the values of offset and spacing are not convertible to ints, or they are too big or too small. Also
        raised, if the remaining pixels are less than 144.
    """
    # Check for different errors
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be of type np.ndarray!")
    if image_array.ndim != 3:
        raise NotImplementedError("image_array must be a 3D array!")
    if image_array.shape[2] != 3:
        raise NotImplementedError("image_array's 3rd dimension must be equal 3!")
    try:
        offset_0, offset_1 = int(offset[0]), int(offset[1])
        spacing_0, spacing_1 = int(spacing[0]), int(spacing[1])
    except ValueError:
        raise ValueError("The values in offset and/or spacing are not convertible to int objects!")
    if not 0 <= offset_0 <= 8 or not 0 <= offset_1 <= 8:
        raise ValueError("One of the values in offset is either too small or too big!")
    if not 2 <= spacing_0 <= 6 or not 2 <= spacing_1 <= 6:
        raise ValueError("One of the values in spacing is either too small or too big!")

    # Create the input_array with zero values and slicing and transpose it
    input_array = np.zeros_like(image_array)
    input_array[offset_1::spacing_1, offset_0::spacing_0] = image_array[offset_1::spacing_1, offset_0::spacing_0]
    input_array = np.transpose(input_array, (2, 0, 1))

    # Create the known_array by coping input_array and slicing
    known_array = np.copy(input_array)
    known_array[:, offset_1::spacing_1, offset_0::spacing_0] = 1
    if np.sum(np.all(known_array > 0, axis=0)) < 144:
        raise ValueError("The number of remaining known pixel values is smaller than 144!")

    image_array = np.transpose(image_array, (2, 0, 1))

    return input_array, known_array, image_array


def to_challenge_target_array(prediction: torch.Tensor, known_array: torch.Tensor) -> np.ndarray:
    """Converts array with datatype float to array with datatype uint8.

    :param prediction: Output of model.
    :param known_array: Array indication which values are known and which are not.
    :return: A 1D array which returns the predictions for the missing pixel values.
    """
    prediction_np: np.ndarray = prediction.cpu().detach().numpy().astype(np.uint8)
    known_array_np: np.ndarray = known_array.cpu().detach().numpy().astype(np.uint8)

    return prediction_np[known_array_np == 0]
