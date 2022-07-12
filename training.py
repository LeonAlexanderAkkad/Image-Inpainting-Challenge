from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer

from helper_methods import get_target_device
from model import ImagePixelPredictor

from typing import Optional, Tuple

import os
from matplotlib import pyplot as plt

from scoring import rmse


def optimizing_predictor(
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        model: ImagePixelPredictor,
        epochs: int,
        loss_function: nn.Module,
        optimizer: Optimizer,
        adapt_lr_factor: Optional[float] = None
        ):
    """Optimizes a given model for a number of epochs and saves the best model.

    The function computes both the MSE (Mean squared error) and the RMSE (Root mean squared error) between
    the output of the model and the given target. Depending on the best RMSE on the validation data, the best
    model is then saved to the specified file.

    Parameters
    ----------
    train_loader: DataLoader
        Data for training the model.
    validation_loader: DataLoader
        Data for monitoring validation loss.
    test_loader: DataLoader
        Data for checking how well the model works on unseen data.
    model: ImagePixelPredictor
        Model to be trained.
    epochs: int
        Number of epochs to train the model for.
    loss_function: nn.Module
        Loss function used to compute the loss between the output of the model and the target.
    optimizer: Optimizer
        Specified optimizer to be used to optimize the model.
    adapt_lr_factor: Optional[float] = None
        Factor used to adapt the learning rate if the model starts to over-fit on the training data.

    Returns
    -------
    None
    """

    best_loss = 0
    lr = get_lr(optimizer)
    writer = SummaryWriter(log_dir=os.path.join("results", "experiment_04"))
    print("\nStarting to train ImagePixelPredictor")
    for epoch in range(epochs):

        train_loss, train_rsme_loss = train_model(model, optimizer, train_loader, loss_function, epoch)
        validation_loss, validation_rsme_loss = eval_model(model, validation_loader, loss_function)

        writer.add_scalar(tag="training/loss",
                          scalar_value=train_loss,
                          global_step=epoch)
        writer.add_scalar(tag="training/rsme_loss",
                          scalar_value=train_rsme_loss,
                          global_step=epoch)
        writer.add_scalar(tag="validation/loss",
                          scalar_value=validation_loss,
                          global_step=epoch)
        writer.add_scalar(tag="validation/rsme_loss",
                          scalar_value=validation_rsme_loss,
                          global_step=epoch)

        print(f"\nEpoch: {str(epoch + 1).zfill(len(str(epochs)))} (lr={lr:.6f} || "
              f"Validation loss: {validation_loss:.4f} | {validation_rsme_loss:.4f} || "
              f"Training loss: {train_loss:.4f} | {train_rsme_loss:.4f})")

        # Either save the best model or adapt the learning rate if necessary.
        if adapt_lr_factor is not None:
            if not best_loss or validation_rsme_loss < best_loss:
                best_loss = validation_rsme_loss
                torch.save(model, "best_model.pt")
                print("Model saved to best_model.pt")
            else:
                lr /= adapt_lr_factor
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                print(f"New learning rate: {lr:.6f}")

        print(100 * "=" + "\n")

    test_loss = eval_model(model, test_loader, loss_function)

    print(f"\nFinal loss: {test_loss}")
    print("\nDone!")


def eval_model(
        model: ImagePixelPredictor,
        test_loader: DataLoader,
        loss_function: nn.Module,
        inspection: bool = False
        ) -> Tuple[float, float]:
    """Evaluates a given model on test data.

    Parameters
    ----------
    model: ImagePixelPredictor
        Model used for evaluation.
    test_loader: DataLoader
        Data used for testing the model.
    loss_function: nn.Module
        Loss function used to determine the "goodness" of the model.
    inspection: bool = False
        Optional parameter used to inspect individual predictions of the model.

    Returns
    -------
    Tuple[float, float]
        A tuple containing both the MSE and RMSE.
    """

    # Turn on evaluation mode for the model.
    model.eval()

    target_device = get_target_device()

    total_loss, total_rmse_loss, num_samples = 0.0, 0.0, 0

    # Compute the loss with torch.no_grad() as gradients aren't used.
    with torch.no_grad():
        for data in test_loader:
            image_array, input_array, known_array = data
            inputs = input_array.type(torch.float32).to(device=target_device)
            known = known_array.type(torch.float32).to(device=target_device)
            targets = image_array.type(torch.float32).to(device=target_device)

            outputs = model(inputs, known)

            # Compute the loss.
            loss = loss_function(outputs, targets)

            # Compute total loss.
            total_loss += loss.item()
            total_rmse_loss += rmse(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())

            # Compute number of samples.
            num_samples += inputs.shape[0]

            # If specified look into individual outputs.
            if inspection:
                inspect_predictions(inputs, outputs, targets)

    return total_loss / num_samples, total_rmse_loss / num_samples


def train_model(
        model: ImagePixelPredictor,
        optimizer: Optimizer,
        training_loader: DataLoader,
        loss_function: nn.Module,
        epoch: int
        ) -> Tuple[float, float]:
    """Trains a given model on the training data.

    Parameters
    ----------
    model: ImagePixelPredictor
        Model to be trained.
    optimizer: Optimizer
        Specified optimizer to be used to optimize the model.
    training_loader: DataLoader
        Data used for training the model.
    loss_function: nn.Module
        Loss function used to compute the loss between the output of the model and the given target.
    epoch: int
        Number of iteration.

    Returns
    -------
    Tuple[float, float]
        A tuple containing both the MSE and RMSE loss.
    """

    target_device = get_target_device()

    # Put the model into train mode and enable gradients computation.
    model.train()
    torch.enable_grad()

    total_loss, total_rmse_loss, num_samples = 0.0, 0.0, 0

    lr = get_lr(optimizer)

    for data in tqdm(training_loader, desc=f"Training epoch {epoch + 1} "
                                           f"(lr={lr:.6f})"):

        image_array, input_array, known_array = data
        inputs = input_array.type(torch.float32).to(device=target_device)
        known = known_array.type(torch.float32).to(device=target_device)
        targets = image_array.type(torch.float32).to(device=target_device)

        outputs = model(inputs, known)

        # Compute loss.
        loss = loss_function(outputs, targets)

        # Compute the gradients.
        loss.backward()

        # Perform the update.
        optimizer.step()

        # Reset the accumulated gradients.
        optimizer.zero_grad()

        # Compute the total loss.
        total_loss += loss.item()
        total_rmse_loss += rmse(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())
        num_samples += inputs.shape[0]

    return total_loss / num_samples, total_rmse_loss / num_samples


def get_lr(optimizer):
    """Get the learning rate used for optimizing."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def inspect_predictions(inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor):
    """Plot individual outputs of the model."""
    input_array_np = inputs[0].to(torch.uint8).permute(1, 2, 0).cpu().detach().numpy()
    output_np = outputs[0].to(torch.uint8).permute(1, 2, 0).cpu().detach().numpy()
    targets_np = targets[0].to(torch.uint8).permute(1, 2, 0).cpu().detach().numpy()

    fig = plt.figure(figsize=(20, 6))

    fig.add_subplot(1, 3, 1)
    plt.imshow(input_array_np)

    fig.add_subplot(1, 3, 2)
    plt.imshow(output_np)

    fig.add_subplot(1, 3, 3)
    plt.imshow(targets_np)

    plt.show()

    input()
