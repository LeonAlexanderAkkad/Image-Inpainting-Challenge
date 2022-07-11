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

    model.eval()

    target_device = get_target_device()

    total_loss, total_rmse_loss, num_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for data in test_loader:
            image_array, input_array, known_array = data
            inputs = input_array.type(torch.float32).to(device=target_device)
            known = known_array.type(torch.float32).to(device=target_device)
            targets = image_array.type(torch.float32).to(device=target_device)

            outputs = model(inputs, known)

            # Compute the loss
            loss = loss_function(outputs, targets)

            # Compute total loss
            total_loss += loss.item()
            total_rmse_loss += rmse(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())

            # Compute number of samples
            num_samples += inputs.shape[0]

            if inspection:
                inspect_predictions(inputs, outputs, targets)

    return total_loss / num_samples, total_rmse_loss / num_samples


def train_model(
        model: ImagePixelPredictor,
        optimizer: Optimizer,
        training_loader: DataLoader,
        loss_function,
        epoch: int
        ) -> Tuple[float, float]:

    target_device = get_target_device()

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

        # Compute loss
        loss = loss_function(outputs, targets)

        # Compute the gradients
        loss.backward()

        # Perform the update
        optimizer.step()

        # Reset the accumulated gradients
        optimizer.zero_grad()

        total_loss += loss.item()
        total_rmse_loss += rmse(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())
        num_samples += inputs.shape[0]

    return total_loss / num_samples, total_rmse_loss / num_samples


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def inspect_predictions(inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor):
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
