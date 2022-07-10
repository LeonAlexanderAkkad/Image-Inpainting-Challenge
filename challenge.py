from helper_methods import get_target_device
from challenge_dataset import ChallengeDataset
from evalutation import compute_predictions

from torch.utils.data import DataLoader
import torch


target_device = get_target_device()

# Create dataset using the class ImageDataset.
image_data = ChallengeDataset()

# Create data_loaders from each subset.
test_loader = DataLoader(
    image_data,
    shuffle=False,
    batch_size=1
)

predictor_model = torch.load("best_model.pt").to(target_device)

compute_predictions(predictor_model, test_loader)
