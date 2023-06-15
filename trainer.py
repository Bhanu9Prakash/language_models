"""
trainer.py
========================
This script configures, trains, and validates a transformer-based language model with PyTorch.
It leverages a preprocessed text dataset, dividing it into training and validation sets.
The model's architecture and training hyperparameters are customizable within the script.
During the training process, model checkpoints are saved at regular intervals.
"""

import torch
from text_data_processor import TextDataProcessor, TextDataset
from model_training import train
from model import LanguageModel
import argparse
from torch.utils.data import DataLoader

# Initialize argument parser
parser = argparse.ArgumentParser(description='Training a language model')

# Add arguments for hyperparameters
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--context_length', type=int, default=32, help='Context length for the transformer model')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device to run the model on')
parser.add_argument('--d_model', type=int, default=64, help='Model dimension parameter for the transformer model')
parser.add_argument('--d_hidden', type=int, default=256, help='Hidden dimension parameter for the transformer model')
parser.add_argument('--n_heads', type=int, default=4, help='Number of heads in the transformer model')
parser.add_argument('--n_layer', type=int, default=4, help='Number of layers in the transformer model')
parser.add_argument('--attn_dropout', type=float, default=0.0, help='Dropout rate for attention in the transformer model')
parser.add_argument('--proj_dropout', type=float, default=0.0, help='Dropout rate for projection in the transformer model')
parser.add_argument('--ffwd_dropout', type=float, default=0.0, help='Dropout rate for feedforward network in the transformer model')

# Add arguments for text preprocessing
parser.add_argument('--file_path', type=str, default='./default_input.txt', help='File path for input text data')
parser.add_argument('--url', type=str, default="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", help='URL for input text data if file path is not available')

# Add arguments for training and saving results
parser.add_argument('--checkpoint_dir', type=str, default="./model_checkpoint", help='Directory to save model checkpoints')
parser.add_argument('--checkpoint_steps', type=int, default=10000, help='Save a checkpoint every these many steps')
parser.add_argument('--save_results_path', type=str, default="./training_results.pth", help='Path to save the training results')

# Parse arguments
args = parser.parse_args()


# model hyperparameters
batch_size = args.batch_size
num_epochs = args.num_epochs
context_length = args.context_length
learning_rate = args.learning_rate
device = args.device
d_model = args.d_model
d_hidden = args.d_hidden
n_heads = args.n_heads
d_head = d_model // n_heads
n_layer = args.n_layer
attn_dropout = args.attn_dropout
proj_dropout = args.proj_dropout
ffwd_dropout = args.ffwd_dropout

# text processing
file_path = args.file_path
url = args.url

# training and save results
checkpoint_dir = args.checkpoint_dir
checkpoint_steps = args.checkpoint_steps
save_results_path = args.save_results_path

# set default device pytorch
torch.set_default_device(device)


processor = TextDataProcessor(file_path = file_path, url = url)
processor.preprocess_text()

train_data, val_data = processor.train_val_split(0.1)

train_dataset = TextDataset(train_data, context_length)
val_dataset = TextDataset(val_data, context_length)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = False)
val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle = False )


# Create model
model = LanguageModel(vocab_size = processor.vocab_size,
                      d_model = d_model,
                      n_layer = n_layer,
                      n_heads = n_heads,
                      d_head = d_head,
                      d_hidden = d_hidden,
                      attn_dropout = attn_dropout,
                      proj_dropout = proj_dropout,
                      ffwd_dropout = ffwd_dropout,
                      context_length = context_length
                      )


model_config = {
    "vocab_size": processor.vocab_size,
    "d_model": d_model,
    "n_layer": n_layer,
    "n_heads": n_heads,
    "d_head": d_head,
    "d_hidden": d_hidden,
    "attn_dropout": attn_dropout,
    "proj_dropout": proj_dropout,
    "ffwd_dropout": ffwd_dropout,
    "context_length": context_length
}



# Create loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                             lr=learning_rate)


# Train model and track results
results = train(model=model,
                                      train_dataloader=train_dataloader,
                                      val_dataloader=val_dataloader,
                                      loss_function=loss_function,
                                      optimizer=optimizer,
                                      checkpoint_dir = checkpoint_dir,
                                      checkpoint_steps = checkpoint_steps,
                                      model_config = model_config,
                                      device = device,
                                      epochs=num_epochs)

torch.save(results, save_results_path)
