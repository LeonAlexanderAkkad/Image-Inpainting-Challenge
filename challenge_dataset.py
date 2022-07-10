from torch.utils.data import Dataset

import dill as pkl

import os


class ChallengeDataset(Dataset):
    def __init__(self):

        with open(os.path.join("test_images", "test_images.pkl"), "rb") as f:
            all_arrays = pkl.load(f)

        self.known_arrays = all_arrays["known_arrays"]
        self.input_arrays = all_arrays["input_arrays"]

    def __getitem__(self, index):
        return self.input_arrays[index], self.known_arrays[index]

    def __len__(self):
        return len(self.input_arrays)
