import torch
from torch.utils.data import DataLoader

from training_dataset import ImageDataset
from model import ImagePixelPredictor
from training import optimizing_predictor
from helper_methods import get_target_device, set_random_seed
from challenge import challenge


target_device = get_target_device()
set_random_seed(42)

# Define loss function.
loss = torch.nn.MSELoss()

# Create dataset using the class ImageDataset.
image_data = ImageDataset("training_images")

# Split the dataset into training-, test- and validation-set using the split method.
training_set, test_set, validation_set = image_data.split(ratios=(5, 1, 1), random_seed=42)

# Create data_loaders from each subset.
test_loader = DataLoader(
    test_set,
    shuffle=False,
    batch_size=32
)
validation_loader = DataLoader(
    validation_set,
    shuffle=False,
    batch_size=32
)
training_loader = DataLoader(
    training_set,
    shuffle=True,
    batch_size=32
)

predictor_model = ImagePixelPredictor(n_input_channels=6,
                                      n_hidden_layers=5,
                                      n_hidden_kernels=64,
                                      n_output_channels=3).to(target_device)

# Define optimizer.
optimizer = torch.optim.Adam(predictor_model.parameters(), lr=1e-2)

# Train model on data.
optimizing_predictor(train_loader=training_loader,
                     validation_loader=validation_loader,
                     test_loader=test_loader,
                     model=predictor_model,
                     epochs=60,
                     loss_function=loss,
                     optimizer=optimizer,
                     adapt_lr_factor=1.2)

result = input("Compute predictions: y / n? ")

if result.lower() == "y":
    challenge()

# TODO: Load model and continue training!
