import _io
import ast

import click
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.classification.accuracy import accuracy
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from chattermill_test.constants import (
    RANDOM_SEED,
    BERT_MODEL_NAME,
    MAX_TOKEN_COUNT,
    LABEL_COLUMNS,
    BATCH_SIZE,
    WARMUP_STEPS,
    TOTAL_TRAINING_STEPS,
    N_EPOCHS
)

#  General settings
pl.seed_everything(RANDOM_SEED)
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, return_dict=True)


# Data loading
def load_data(data_file):
    df = pd.read_csv(data_file)
    df.drop(df.tail(1).index, inplace=True)

    # Label data binarization
    df_aspects = [ast.literal_eval(i) for i in df.aspects]
    ids, label_names = pd.factorize(np.concatenate(df_aspects))
    df_out = pd.DataFrame([np.isin(label_names, i) for i in df_aspects], columns=label_names).astype(int)
    df = df.join(df_out)

    #  Data splitting into training and testing datasets
    train_df, val_df = train_test_split(df, test_size=0.05)
    train_df, test_df = train_test_split(train_df, test_size=0.1)
    return train_df, val_df, test_df


class CommentsDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: BertTokenizer,
            max_token_len: int = MAX_TOKEN_COUNT
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        comment = data_row.comment
        labels = data_row[LABEL_COLUMNS]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return dict(
            comment=comment,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


# train_df = load_data("/home/taras/PycharmProjects/chattermill/nlp-challenge-master/train.csv")
# train_dataset = CommentsDataset(
#     train_df,
#     tokenizer,
#     max_token_len=MAX_TOKEN_COUNT
# )

class CommentDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, val_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = CommentsDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = CommentsDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

        self.val_dataset = CommentsDataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=12
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=12
        )


class CommentTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def _step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        return loss, outputs, labels

    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(LABEL_COLUMNS):
            class_accuracy = accuracy(predictions[:, i], labels[:, i])
            # class_roc_auc = auroc(predictions[:, i], labels[:, i])
            # self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
            self.logger.experiment.add_scalar(f"{name}_accuracy/Train", class_accuracy, self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


@click.command()
@click.option('--input_file',
              type=click.File("rb"),
              prompt='Specify the input csv file',
              help='Enter the path to the file with the training data.')
def train(input_file: _io.BufferedReader):
    train_df, val_df, test_df = load_data(input_file)

    # Training optimization
    dummy_model = nn.Linear(2, 1)
    optimizer = AdamW(params=dummy_model.parameters(), lr=0.001)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_TRAINING_STEPS
    )
    learning_rate_history = []
    for step in range(TOTAL_TRAINING_STEPS):
        optimizer.step()
        scheduler.step()
        learning_rate_history.append(optimizer.param_groups[0]['lr'])
    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS

    warmup_steps = total_training_steps // 5
    warmup_steps, total_training_steps

    # TensorBoard logging
    logger = TensorBoardLogger("lightning_logs", name="comments")

    # Trainer settings
    model = CommentTagger(
        n_classes=len(LABEL_COLUMNS),
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    data_module = CommentDataModule(
        train_df,
        test_df,
        val_df,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_COUNT
    )

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=True,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
    )
    trainer.fit(model, data_module)
    trainer.test()


if __name__ == '__main__':
    train()
