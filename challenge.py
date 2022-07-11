from helper_methods import get_target_device
from challenge_dataset import ChallengeDataset
from evalutation import compute_predictions

from torch.utils.data import DataLoader
import torch

import os


def challenge():

    target_device = get_target_device()

    # Create dataset using the class ChallengeDataset.
    image_data = ChallengeDataset(os.path.join("test_images", "test_images.pkl"))

    # Create data_loaders from test_set.
    test_loader = DataLoader(
        image_data,
        shuffle=False,
        batch_size=1
    )

    # Load trained model.
    predictor_model = torch.load("best_model.pt").to(target_device)

    # Compute predictions.
    compute_predictions(predictor_model, test_loader)


if __name__ == "__main__":
    challenge()
