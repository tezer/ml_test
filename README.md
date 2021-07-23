# ml_test

## Update
**I added experiments with _Simple transformers_ + _wandb_: `run_classifier_st.py` and `train_classifier.py`. 
The output is `test_answers_st.csv`.**

**_Simple transformers_ add even more simplicity to training code, while _wandb_ brings visualization hyperparameter tuning.**
**Although the _wandb_ would work with other frameworks too (e.g. PyTorch, TensorFlow, Keras), I like how Simple _Transformers integrate_ with _wandb_**

**To train a model with _Simple transformers_ + _wandb_ you need an account at <https://wandb.ai/>**

**end of update**

This is a simple command-line utility for multi-label text classification. There is a trainer (`train_classifier.py`)
and a text tagger (`run_classifier.py`). The file `constants.py`contains the most important settings for training and
for tagging.

I used BERT `bert-base-cased` for this task as the most appropriate type of transformer. There could be other similar
alternatives, but I didn't have time and hardware to experiment with them.

The main libraries used in the project are: `transformers`, `pandas`, `torch` and `pytorch_lightning`. The latter makes
the PyTorch-based code more readable.

Here is what the code does for training:

1. loads the data to DataFrame and binarizes the labels
1. splits the dataset into train, test and validation subsets
1. defines a scheduler to change the learning rate of the optimizer (to improve performance of the mode)
1. defines two callbacks: one for early stopping that tracks validation loss, and best checkpoint saver to make sure I
   have the best trained model
1. finally, packs everything into the Trainer and fires it off

run_classifier.py is much simpler:

1. it reads the data file and tokenizes the texts
1. loads the saved model (a checkpoint file)
1. gets predictions (tags) for each text from the model
1. and writes them to the specified file

## Installation

clone the project with `git clone https://github.com/tezer/ml_test

## Usage

Requires python 3.8
`cd ml_test`

For model training run:

`poetry run python chuttermill/train_classifier.py` for interactive mode

or

```
poetry run python train_classifier.py --input_file=/path/to/your/train.csv
```

where

```
Options:
  --input_file FILENAME   Enter the path to the file to be processed.
  --help                  Show this message and exit.
```

For text tagging run:

`poetry run python chuttermill/run_classifier.py` for interactive mode

or

```
poetry run python run_classifier.py --input_file=/path/to/your/test.csv
--model_file=/path/to/your/best-checkpoint.ckpt --output_file=/path/to/result.csv (optional, default value is test_answers.csv)
```

where

```
Options:
  --input_file FILENAME   Enter the path to the csv file with the comments to be tagged.
  --model_file FILENAME   Enter the path to the file with the trained model.
  --output_file FILENAME  You need to specify a file name where the output will be saved. The default value is test_answers.csv
  --help                  Show this message and exit.
