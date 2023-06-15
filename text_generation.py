"""
text_generation.py
========================
This script loads a trained transformer-based language model from a PyTorch checkpoint,
then generates and prints text based on provided input.
It also gives an option to save the generated text to a file.
"""

import argparse
import torch
from model import LanguageModel
from text_data_processor import TextDataProcessor

# Initialize argument parser
parser = argparse.ArgumentParser(description='Text generation with a trained model')

# Add arguments
parser.add_argument('--checkpoint', type=str, default = "./training_results.pth", help='Path to the model checkpoint')
parser.add_argument('--input_text', type=str, required=True, help='Input text to feed the model')
parser.add_argument('--max_input_tokens', type=int, default=100, help='The maximum number of input tokens')
parser.add_argument('--save_output', type=str, default="./output.txt", help='File path to save the generated text')

# Parse arguments
args = parser.parse_args()

# Load model state from checkpoint
checkpoint = torch.load(args.checkpoint)

# Instantiate the model with the saved configuration
model = LanguageModel(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Assuming that the model is on the same device that was used for saving
# Move the model to evaluation mode
model.eval()

# Instantiate text processor
processor = TextDataProcessor()
processor.preprocess_text()
# Convert input text to tensor and generate text
input_tensor = torch.tensor(processor.encode(args.input_text), dtype=torch.long).unsqueeze(0)
generated_text_tensor = model.generate(input_tensor, args.max_input_tokens)[0].tolist()

# Decode generated tensor to text
output = processor.decode(generated_text_tensor)

# Print generated text
print("Generated text:\n", output)

# Save generated text to file if save path is provided
if args.save_output:
    with open(args.save_output, 'w') as f:
        f.write(output)
