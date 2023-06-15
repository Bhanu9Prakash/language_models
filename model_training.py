"""
model_training.py
=================

A module to train and validate a PyTorch model using specified DataLoaders. This module contains functions to perform training and validation steps (epochs), as well as timing these steps.

Key functionalities are:
1. `train_step`: A function that performs a single training step (epoch), involving running the model on a batch of input data, computing the loss, performing backpropagation, and updating the model parameters.
2. `val_step`: A function that performs a single validation step (epoch), where the model is run on the validation dataset to compute the validation loss. The model parameters are not updated in this step.
3. `train`: This function loops over the specified number of epochs, calling the `train_step` and `val_step` functions for each epoch. It computes and prints out the training and validation losses, and the time taken for each epoch.

The results from the training and validation steps are returned in a dictionary. This script is particularly useful for training deep learning models using PyTorch and monitoring their performance over time.

"""

import time
import os
import copy
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union

def train_step(epoch:int, model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_function:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               epoch_dir:str,
               checkpoint_steps:int,
               device:torch.device,
               model_config:Dict[str, Union[int, float]],
               disable_progress_bar: bool = False) -> Tuple[float, float]:

  model.train()
  train_loss = 0.0
  # Loop through data loader data batches
  progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar
    )


  for batch, (X, y) in progress_bar:
      X, y = X.to(device), y.to(device)
      logits = model(X)
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      y = y.view(B*T)
      loss = loss_function(logits, y)
      train_loss += loss.item()

      # backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Checkpoint saving condition
      if (batch+1) % checkpoint_steps == 0:
        torch.save({
            'epoch': epoch+1,
            'step': batch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'model_config':model_config,
        }, os.path.join(epoch_dir, f'checkpoint_step_{batch+1}.pt'))

       # Update progress bar
      progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
            }
        )
  train_loss /= len(dataloader)

  return train_loss

def val_step(epoch:int, model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_function:torch.nn.Module,
               device:torch.device,
               disable_progress_bar: bool = False) -> Tuple[float, float]:

  model.eval()
  val_loss = 0.0
  # Loop through data loader data batches
  progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Validating Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

  with torch.no_grad():
    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        loss = loss_function(logits, y)
        val_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(
              {
                  "val_loss": val_loss / (batch + 1),
              }
          )
    val_loss /= len(dataloader)

  return val_loss

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_function: torch.nn.Module,
          epochs: int,
          checkpoint_dir:str,
          checkpoint_steps:int,
          model_config:Dict[str, Union[int, float]],
          device:torch.device,
          disable_progress_bar: bool = False) -> Dict[str, List]:

  best_loss = float('inf')

  # Create empty results dictionary
  results = {"train_loss": [],
      "val_loss": [],
      "train_epoch_time": [],
      "val_epoch_time": [],
      "model_state_dict":None,
      "model_config": model_config
  }

  # Create checkpoint directory if it doesn't exist
  if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

# Loop through training and valing steps for a number of epochs
  for epoch in tqdm(range(epochs), disable=disable_progress_bar):

      # Create a directory for the epoch if it doesn't exist
      epoch_dir = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
      if not os.path.exists(epoch_dir):
          os.makedirs(epoch_dir)

      # Perform training step and time it
      train_epoch_start_time = time.time()
      train_loss = train_step(epoch=epoch,
                                        model=model,
                                        dataloader=train_dataloader,
                                        loss_function=loss_function,
                                        optimizer=optimizer,
                                        device = device,
                                        epoch_dir = epoch_dir,
                                        checkpoint_steps = checkpoint_steps,
                                        model_config = model_config,
                                        disable_progress_bar=disable_progress_bar)
      train_epoch_end_time = time.time()
      train_epoch_time = train_epoch_end_time - train_epoch_start_time

      # Perform valing step and time it
      val_epoch_start_time = time.time()
      val_loss = val_step(epoch=epoch,
                          model=model,
                          dataloader=val_dataloader,
                          loss_function=loss_function,
                          device = device,
                          disable_progress_bar=disable_progress_bar)
      val_epoch_end_time = time.time()
      val_epoch_time = val_epoch_end_time - val_epoch_start_time

      if val_loss < best_loss:
        best_loss = val_loss
        results["model_state_dict"] = copy.deepcopy(model.state_dict())

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"train_epoch_time: {train_epoch_time:.4f} | "
          f"val_epoch_time: {val_epoch_time:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["val_loss"].append(val_loss)
      results["train_epoch_time"].append(train_epoch_time)
      results["val_epoch_time"].append(val_epoch_time)

  return results
