import _io
import csv
from typing import List, Tuple

import click
from simpletransformers.classification import MultiLabelClassificationModel

from chattermill_test.constants import LABEL_COLUMNS

THRESHOLD = .3


def tag_data(csv_file, model_folder) -> List[Tuple[str, str]]:
    result = []
    comments = []
    with csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            comments.append(row[0])
    model = MultiLabelClassificationModel(
        "bert", model_folder
    )
    # Make predictions with the model
    _, predictions = model.predict(comments)
    for i, prediction in enumerate(predictions):
        predicted_labels = [label for label, prediction in zip(LABEL_COLUMNS, prediction) if
                            prediction > THRESHOLD]
        result.append([comments[i], str(predicted_labels)])
    return result


def write_data(data: List[Tuple[str, str]], outfile_path: str):
    header = ['comment', 'aspects']
    with open(outfile_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


@click.command()
@click.option('--input_file',
              type=click.File("r"),
              prompt='Specify the input csv file',
              help='Enter the path to the csv file with the comments to be tagged.')
@click.option('--model_folder',
              type=str,
              prompt='Specify the model folder (default: outputs/best_model)',
              default="outputs/best_model",
              help='Enter the path to the folder with the trained model data.')
@click.option(
    '--output_file',
    type=str,
    default="test_answers.csv",
    prompt='Specify the output file (default is test_answers.csv)',
    help=
    'You need to specify a file name where the output will be saved. The default value is test_answers.csv'
)
def start(input_file: _io.BufferedReader, model_folder: str, output_file: str):
    data: List[Tuple[str, str]] = tag_data(input_file, model_folder)
    write_data(data, output_file)


if __name__ == '__main__':
    start()
