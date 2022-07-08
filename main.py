from dataset import ImageDataset
from model import ImagePixelPredictor
from helper_methods import get_target_device, set_random_seed
from training import train_nn


target_device = get_target_device()
set_random_seed(69)

data = ImageDataset("training_images")

data_train, data_test, data_validation = data.split(ratios=(6, 1, 1), random_seed=69)

