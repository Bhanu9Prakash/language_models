"""
text_data_processor.py
======================

A Python module that contains classes for processing and managing text data for deep learning models. 
It provides functionality to load text data from various sources such as single/multiple URLs, local files, or entire directories. 

It includes the following main classes:
1. TextDataProcessor: The base class that handles the common operations such as text preprocessing, 
   splitting the data into training and validation sets, and encoding and decoding the text data. This class is intended to be subclassed for specific data source types.
2. URLTextDataProcessor: Subclass of TextDataProcessor that loads text data from one or multiple URLs, or a file containing multiple URLs.
3. FileTextDataProcessor: Subclass of TextDataProcessor that loads text data from local file(s).
4. DirectoryTextDataProcessor: Subclass of TextDataProcessor that loads text data from all text files in a directory.
5. TextDataset: A PyTorch Dataset class for creating context-target pairs from a list of integers. 
   It can be used to feed data into a sequence prediction model such as an LSTM or Transformer.
Each of these classes includes detailed method-level comments describing their functionality, arguments, and return values. 

This module can be used as a preprocessing script for tasks such as language modelling, text generation, and other natural language processing tasks.
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Union
import psutil
from tqdm import tqdm

class TextDataProcessor:
  """
  A base class for downloading, reading, preprocessing, and splitting text data.

  Args:
    source (Union[str, List[str]]): Source of the data, can be a string or a list of strings depending on the specific subclass used.

  Attributes:
    source (Union[str, List[str]]): Source of the data.
    chars (List[char]): Unique characters in the dataset.
    stoi (Dict): Dictionary to convert a character to its corresponding integer representation.
    itos (Dict): Dictionary to convert an integer back to its corresponding character.
    text (str): The raw text data loaded.
    vocab_size (int): The size of the vocabulary (i.e., number of unique characters in the dataset).
  """

  def __init__(self, source: Union[str, List[str]]):
    self.source = source
    self.chars = []
    self.stoi = {}
    self.itos = {}
    self.text = self.load_data()
    self.vocab_size = 0
    self.preprocess_text()

  def load_data(self):
    """
    This is a placeholder function. It should be implemented in subclasses.
    """
    raise NotImplementedError

  def preprocess_text(self) -> None:
    """
    Processes the raw text data. Constructs mappings from characters to integers and vice versa. 
    Also computes the size of the vocabulary.
    """
    self.chars = sorted(set(self.text))
    self.stoi = {ch: i for i, ch in enumerate(self.chars)}
    self.itos = {i: ch for i, ch in enumerate(self.chars)}
    self.vocab_size = len(self.chars)

  def encode(self, s:str) -> List[int]:
    """
    Encodes a string into a list of integers using the character-to-integer mapping.

    Args:
      s (str): The string to encode.

    Returns:
      List[int]: The encoded string as a list of integers.
    """
    return [self.stoi[ch] for ch in s]

  def decode(self, l:List[int]) -> str:
    """
    Decodes a list of integers into a string using the integer-to-character mapping.

    Args:
      l (List[int]): The list of integers to decode.

    Returns:
      str: The decoded list of integers as a string.
    """
    return ''.join(self.itos[i] for i in l)

  def train_val_split(self, split_ratio:float = 0.1) -> Tuple[List[int], List[int]]:
    """
    Splits the data into training and validation sets.

    Args:
      split_ratio (float, optional): The fraction of the data to be used as the validation set. Default is 0.1.

    Returns:
      Tuple[List[int], List[int]]: The encoded training and validation sets as lists of integers.
    """
    n = int(len(self.text)*(1-split_ratio))
    train_data = self.text[:n]
    val_data = self.text[n:]
    return self.encode(train_data), self.encode(val_data)


class URLTextDataProcessor(TextDataProcessor):
  """
  A TextDataProcessor subclass for handling data from URLs.

  Args:
    source (Union[str, List[str]]): Source of the data, can be a string (single URL) or a list of strings (multiple URLs).
    is_url_file (bool): If set to True, source is assumed to be a filepath to a file containing URLs. Each line in the file should be a separate URL. 
                        If set to False, source is assumed to be one or more direct URLs. 
  """

  def __init__(self, source: Union[str, List[str]], is_url_file: bool = False):
    self.is_url_file = is_url_file
    super().__init__(source)

  def load_data(self):
    """
    Loads data from the provided URLs or from a file containing URLs. 
    It reads the data in chunks to manage memory usage.

    Returns:
      str: The raw text data as a string.
    """
    text = ""
    urls = []
    if isinstance(self.source, str):
      if self.is_url_file:
        with open(self.source, 'r') as f:
          urls += [line.strip() for line in f.readlines()]
      else:
        urls = [self.source] # single url
    else:
      urls = self.source

    for url in urls:
      response = urllib.request.urlopen(url)
      read_size = psutil.virtual_memory().available // 2**3  # Read at max 1/8th of available memory
      while True:
        chunk = response.read(read_size)
        if not chunk:
            break
        text += chunk.decode('utf-8', errors='ignore')
    return text



class FileTextDataProcessor(TextDataProcessor):
  """
  A TextDataProcessor subclass for handling data from local files.

  Args:
    source (Union[str, List[str]]): Source of the data, can be a single file path or a list of file paths.
  """
  def __init__(self, source: Union[str, List[str]]):
      if isinstance(source, str):
          source = [source]
      super().__init__(source)

  def load_data(self):
    """
    Loads data from the provided local files.

    Returns:
      str: The raw text data as a string.
    """
    text = ""
    for file_path in self.source:
      with open(file_path, mode='r', encoding='utf-8') as f:
        text += f.read()
    return text


class DirectoryTextDataProcessor(TextDataProcessor):
  """
  A TextDataProcessor subclass for handling data from directories.
  Assumes that the directory contains .txt files.

  Args:
      source (str): Source of the data, should be a directory path.
  """

  def load_data(self):
    text = ""
    for filename in tqdm(os.listdir(self.source), desc='Reading files'):
      if filename.endswith('.txt'):
        with open(os.path.join(self.source, filename), mode='r', encoding='utf-8') as f:
          text += f.read()
    return text


class TextDataset(Dataset):
  """
  A PyTorch Dataset for creating context-target pairs from a list of integers.

  Args:
    data (List[int]): The text data as a list of integers.
    context_length (int): The length of the context used for prediction.

  Attributes:
    data (List[int]): The text data as a list of integers.
    context_length (int): The length of the context used for prediction.
  """
  def __init__(self, data:List[int], context_length:int) -> None:
      super().__init__()
      self.data = data
      self.context_length = context_length

  def __len__(self) -> int:
    """
    Returns the number of context-target pairs.

    Returns:
      int: The number of context-target pairs.
    """
    return len(self.data) - self.context_length

  def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(self.data[idx : idx + self.context_length], dtype = torch.long)
    y = torch.tensor(self.data[idx + 1 : idx + 1 + self.context_length], dtype = torch.long)
    return x, y

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
