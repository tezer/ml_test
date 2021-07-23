import _io
import ast
import logging

import click
import numpy as np
import pandas as pd
import wandb
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)
from sklearn.model_selection import train_test_split

PROJECT_ID = "chuttermill"

# wandb.login(key="YOUR_WANDB_KEY")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

LABEL_NAMES = []
train_df, eval_df, test_df = None, None, None

# model configuration
model_args = MultiLabelClassificationArgs()
#  Early stopping
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "eval_loss"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 1000
# Training settings
model_args.manual_seed = 4
model_args.use_multiprocessing = True
model_args.train_batch_size = 4
model_args.eval_batch_size = 4
# Other
model_args.overwrite_output_dir = True
model_args.wandb_project = PROJECT_ID
model_args.reprocess_input_data = True


# Data loading
def load_data(data_file):
    df = pd.read_csv(data_file)
    df.drop(df.tail(1).index, inplace=True)

    # Label data binarization
    df_aspects = [ast.literal_eval(i) for i in df.aspects]
    ids, label_names = pd.factorize(np.concatenate(df_aspects))
    df_out = pd.DataFrame([np.isin(label_names, i) for i in df_aspects], columns=label_names).astype(int)
    label_names = df_out.columns.tolist()
    df["labels"] = df_out.values.tolist()
    # df = df.join(df_out)
    df.drop(['aspects'], axis=1, inplace=True)
    df.columns = ["text", "labels"]
    #  Data splitting into training and testing datasets
    train_df, val_df = train_test_split(df, test_size=0.05)
    train_df, test_df = train_test_split(train_df, test_size=0.1)
    return train_df, val_df, test_df, label_names


def train():
    wandb.init()
    # Create a ClassificationModel
    model = MultiLabelClassificationModel(
        'bert',
        'bert-base-cased',
        num_labels=len(LABEL_NAMES),
        args=model_args,
        sweep_config=wandb.config,
    )

    # Train the model
    model.train_model(train_df, eval_df=eval_df)

    # Evaluate the model
    result, _, _ = model.eval_model(eval_df)
    wandb.join()


@click.command()
@click.option('--input_file',
              type=click.File("r"),
              prompt='Specify the input csv file',
              help='Enter the path to the file with the training data.')
def start(input_file: _io.BufferedReader):
    # Config for hyperparameters optimization
    sweep_config = {
        "method": "bayes",  # grid, random
        "metric": {"name": "train_loss", "goal": "minimize"},
        "parameters": {
            "num_train_epochs": {"values": [2, 3, 5]},
            "learning_rate": {"min": 5e-5, "max": 4e-4},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_ID)
    global train_df, eval_df, test_df, LABEL_NAMES
    train_df, eval_df, test_df, LABEL_NAMES = load_data(input_file)
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    start()
