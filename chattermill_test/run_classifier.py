import _io
import csv
from typing import List, Tuple

import click
from tqdm.auto import tqdm

from chattermill_test.constants import LABEL_COLUMNS, MAX_TOKEN_COUNT, THRESHOLD
from chattermill_test.train_classifier import tokenizer, CommentTagger


def tag_data(csv_file: _io.BufferedReader, model_path: _io.BufferedReader) -> List[Tuple[str, str]]:
    result = []
    trained_model = CommentTagger.load_from_checkpoint(
        model_path,
        n_classes=len(LABEL_COLUMNS)
    )
    trained_model.eval()
    trained_model.freeze()
    comments = []
    with csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            comments.append(row[0])
    for comment in tqdm(comments[1:]):  # Skipping the header
        encoding = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=MAX_TOKEN_COUNT,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        _, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
        test_prediction = test_prediction.flatten().numpy()
        predicted_labels = [label for label, prediction in zip(LABEL_COLUMNS, test_prediction) if
                            prediction > THRESHOLD]
        result.append([comment, str(predicted_labels)])
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
@click.option('--model_file',
              type=click.File("rb"),
              prompt='Specify the model file',
              help='Enter the path to the file with the trained model.')
@click.option(
    '--output_file',
    type=str,
    default="test_answers.csv",
    prompt='Specify the output file (default is test_answers.csv)',
    help=
    'You need to specify a file name where the output will be saved. The default value is test_answers.csv'
)
def start(input_file: _io.BufferedReader, model_file: _io.BufferedReader, output_file: str):
    data: List[Tuple[str, str]] = tag_data(input_file, model_file)
    write_data(data, output_file)


if __name__ == '__main__':
    start()
