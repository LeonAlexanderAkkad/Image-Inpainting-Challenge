from torch.utils.data import Dataset

import dill as pkl


class ChallengeDataset(Dataset):
    """Dataset for the challenge data."""
    def __init__(self, file_path):
        """Initializes the dataset by loading the data from the specified path.

        Parameters
        ----------
        file_path: str
            Path to data.
        """

        with open(file_path, "rb") as f:
            all_arrays = pkl.load(f)

        self.known_arrays = all_arrays["known_arrays"]
        self.input_arrays = all_arrays["input_arrays"]

    def __getitem__(self, index):
        return self.input_arrays[index], self.known_arrays[index]

    def __len__(self):
        return len(self.input_arrays)
