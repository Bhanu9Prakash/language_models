"""
text_data_processor.py
======================

A module to download, preprocess, and load text data for character-level language modeling.
This module contains two classes, TextDataProcessor and TextDataset, which can be used for processing and loading text data.

"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset
from typing import Tuple, List

class TextDataProcessor:
  """
  A class to download, read, preprocess, and split text data.

  Args:
      file_path (str): Path to the file where the data should be saved.
      url (str): URL to download the data.
  """
  def __init__(self, file_path: str = './default_input.txt',
                 url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"):
    self.file_path = file_path
    self.url = url
    self.chars = []
    self.stoi = {}
    self.itos = {}
    self.text = self.get_data()
    self.vocab_size = 0

  def get_data(self):
    """
    Downloads the data from the url if it doesn't exist at file_path, and reads the data from the file.

    Returns:
        str: Text data.
    """
    # Check if the file already exists
    if not os.path.exists(self.file_path):
        # File doesn't exist, so download it
      urllib.request.urlretrieve(self.url, self.file_path)


    # Read the file
    with open(self.file_path, mode='r', encoding='utf-8') as f:
        text = f.read()

    return text

  def preprocess_text(self) -> None:
    """
    Identifies unique characters in the text, creates encoding and decoding dictionaries,
    and sets vocabulary size.
    """
    self.chars = sorted(set(self.text))
    self.stoi = {ch:i for i, ch in enumerate(self.chars)}
    self.itos = {i:ch for i, ch in enumerate(self.chars)}
    self.vocab_size = len(self.chars)

  def encode(self, s:str) -> List[int]:
    """
    Encodes the given string into integers using character-to-integer dictionary.

    Args:
        s (str): String to be encoded.

    Returns:
        List[int]: Encoded string.
    """
    return [self.stoi[ch] for ch in s]

  def decode(self, l:List[int]) -> str:
    """
    Decodes the given list of integers into string using integer-to-character dictionary.

    Args:
        l (List[int]): List of integers to be decoded.

    Returns:
        str: Decoded string.
    """
    return ''.join(self.itos[i] for i in l)

  def train_val_split(self, split_ratio:float = 0.1) -> Tuple[List[int], List[int]]:
    """
    Splits the data into training and validation parts.

    Args:
        data (Tensor): Data to be split.
        split_ratio (float): The ratio of training data to total data. Default is 0.9.

    Returns:
        Tuple[Tensor, Tensor]: Training data and validation data.
    """
    n = int(len(self.text)*(1-split_ratio))
    train_data = self.text[:n]
    val_data = self.text[n:]
    return self.encode(train_data), self.encode(val_data)

class TextDataset(Dataset):
  """
  A PyTorch Dataset for character-level language modeling.

  Args:
      data (Tensor): The encoded text data.
      context_length (int): The length of the sequence to be considered for language modeling.
  """
  def __init__(self, data:str, context_length:int) -> None:
    super().__init__()
    self.data = data
    self.context_length = context_length

  def __len__(self) -> int:
    """Returns the length of the data."""
    return len(self.data) - self.context_length

  def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple of input data and target data.

    Args:
        idx (int): Index.

    Returns:
        Tuple[Tensor, Tensor]: Tuple of input data and target data.
    """
    x = torch.tensor(self.data[idx : idx + self.context_length], dtype = torch.long)
    y = torch.tensor(self.data[idx + 1 : idx + 1 + self.context_length], dtype = torch.long)
    return x, y
