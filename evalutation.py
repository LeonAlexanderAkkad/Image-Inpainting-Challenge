from model import ImagePixelPredictor
from helper_methods import get_target_device, to_challenge_target_array

from torch.utils.data import DataLoader
import torch

import dill as pkl


def compute_predictions(model: ImagePixelPredictor,
                        test_loader: DataLoader
                        ) -> None:
    """Compute the model predictions and store them in a file.

    Parameters
    ----------
    model: ImagePixelPredictor
        Model used to compute the predictions.
    test_loader: DataLoader
        Test data used to compute the predictions.

    Returns
    -------
    None
    """
    model.eval()

    target_device = get_target_device()

    predictions = []

    with torch.no_grad():
        for data in test_loader:
            input_array, known_array = data
            inputs = input_array.type(torch.float32).to(device=target_device)
            known = known_array.type(torch.float32).to(device=target_device)

            outputs = model(inputs, known)

            predictions.append(to_challenge_target_array(outputs, known))

        with open("predictions.pkl", "wb") as f:
            pkl.dump(predictions, f)
