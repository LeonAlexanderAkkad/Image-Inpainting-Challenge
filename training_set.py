import numpy as np
from PIL import Image
import os
import random
import dill as pkl
from helper_methods import ex4

import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
import glob


class TrainingDataset(Dataset):
    def __init__(self, sequence_length: int, n_features: int):
        """
        """

    def __getitem__(self, index):
        """
        """

    def __len__(self):
        return self.n_samples

    ####################################################
    # Helper Methods
    ####################################################

    def __load_dataset(self, file_path: str):
        content = glob.glob(os.path.join(file_path, '**', '*'), recursive=True)
        resize = torchvision.transforms.Resize((100, 100), interpolation=InterpolationMode.BILINEAR)
        random.seed(69)
        all_images = {}
        image_arrays, known_arrays, input_arrays = [], [], []

        for picture in content:
            with Image.open(picture) as image:
                image = resize(image)
                image_np = np.array(image)
                offset_0 = random.randint(0, 8)
                offset_1 = random.randint(0, 8)
                spacing_0 = random.randint(2, 6)
                spacing_1 = random.randint(2, 6)
                input_array, known_array, image_array = ex4(image_np, (offset_0, offset_1), (spacing_0, spacing_1))
                input_arrays.append(image_array)
                known_arrays.append(known_array)
                image_arrays.append(image_array)

        all_images["input_arrays"] = input_arrays
        all_images["known_arrays"] = known_arrays
        all_images["image_arrays"] = image_arrays

        with open("images.pkl", "wb") as f:
            pkl.dump(all_images, f)
