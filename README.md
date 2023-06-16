# Transformer-Based Language Model Training and Generation

This repository contains Python scripts for training a transformer-based language model and using the trained model to generate text. The scripts are built with PyTorch and allow you to configure model hyperparameters, preprocess text data, and control how model checkpoints are saved during training.

## Getting Started

1. Clone the repository:
    ```
    git clone https://github.com/Bhanu9Prakash/language_models.git
    cd language_models
    ```
2. Install the required libraries:
    ```
    pip install -r requirements.txt
    ```

## Training the Model

The `trainer.py` script is used for training the language model. The script leverages a preprocessed text dataset and divides it into training and validation sets. The model's architecture and training hyperparameters are customizable via command-line arguments. During the training process, model checkpoints are saved at regular intervals. 

After preprocessing, the text data is converted to a PyTorch TextDataset which generates pairs of context and target sequences for training the language model. The context is a sequence of n characters, and the target is the next character to be predicted.

Here is an example command to run the script:

```bash
python trainer.py --batch_size 32 --num_epochs 2 --file_path "./new_input.txt" --checkpoint_dir "./new_checkpoint_dir" --save_results_path "./new_training_results.pth"
```

The above command will start training the model with a batch size of 32, for 2 epochs. It uses the text data from the file `new_input.txt` for training. Model checkpoints will be saved in the directory `new_checkpoint_dir`. The training results (final model state and configurations) will be saved at `new_training_results.pth`.

You can view all available options by running `python trainer.py --help`.

## Text Data Processors
An important aspect of the training process is the preprocessing of the input text data. We use the TextDataProcessor class (and its subclasses URLTextDataProcessor, FileTextDataProcessor, DirectoryTextDataProcessor) from the text_data_processor.py module to handle this. These classes load text data from various sources, preprocess it by encoding characters as integers, and split it into training and validation sets.

For example, if you have a text file stored locally, you could use the FileTextDataProcessor to load and preprocess the text data for training. If you have multiple text files in a directory, you could use the DirectoryTextDataProcessor to process all text files at once. Alternatively, if your text data is located online, the URLTextDataProcessor can be used to download and preprocess the data.



## Generating Text

The `text_generation.py` script is used for generating text with the trained model. This script loads a trained transformer-based language model from a PyTorch checkpoint, then generates and prints text based on provided input. It also gives an option to save the generated text to a file.

Here is an example command to run the script:

```bash
python text_generation.py --checkpoint path_to_checkpoint.pth --input_text "Your input text" --max_input_tokens 100 --save_output generated_text.txt
```

The above command will load the model from `path_to_checkpoint.pth`, generate text based on the "Your input text", and save the generated text to `generated_text.txt`.

You can view all available options by running `python text_generation.py --help`.

## Author

Bhanu Prakash

## License

This project is licensed under the MIT License - see the LICENSE file for details.
