from importlib.util import set_loader
import pytorch_lightning as pl
import pandas as pd
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import transformers
from transformers import AutoTokenizer, AutoModel

class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.tokenizer = tokenizer
        self.df = df
        self.classes = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']
        self.max_len = max_len
        self.is_test = is_test

    def _preprocess(self, text):
        text = text.lower()
        # remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # remove extra spacesf
        text = re.sub(r'\s+', ' ', text)
        # remove non-roman characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.full_text[index]
        text = self._preprocess(text)
      

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        inputs =  {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }
        if self.is_test:
            return inputs

        label = self.df.loc[index,self.classes].to_list()
        targets =  torch.tensor(label, dtype=torch.float32)

        return inputs, targets
        


class Model(pl.LightningModule):
    def __init__(self,CFG):
        super().__init__()
        self.CFG = CFG
        self.loss_func = nn.SmoothL1Loss()
        self.train_df = pd.read_csv(self.CFG["train_path"])
        self.test_df = pd.read_csv(self.CFG["test_path"])
        self.tokenizer  = AutoTokenizer.from_pretrained(self.CFG["model_name"])
        self.encoder = AutoModel.from_pretrained(self.CFG["model_name"])
        self.dropout = nn.Dropout(self.CFG["dropout"])
        self.out = nn.Linear(self.encoder.config.hidden_size,self.CFG["num_classes"])

       

    # def _loss_fn(self, outputs, targets):
    #     colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
    #     loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    #     return loss

    def forward(self, x):
        #encoder returns inputs and outputs
        _, outputs = self.encoder(input_ids=x['input_ids'], attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'], return_dict=False)
        outputs = self.dropout(outputs)
        outputs = self.out(outputs)
        return outputs


    def train_dataloader(self):
        ds = FeedbackDataset(self.train_df, self.tokenizer,self.CFG["max_len"])
        return DataLoader(ds, batch_size=self.CFG["train_batch_size"], shuffle=True)

    def test_dataloader(self):
        ds = FeedbackDataset(self.test_df, self.tokenizer, self.CFG["max_len"], is_test=True)
        return DataLoader(ds, batch_size=self.CFG["val_batch_size"], shuffle=False)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        outputs = self(inputs)
        loss = self.loss_func(outputs, targets)
        self.log("train_loss", loss)
        return loss

   
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.loss_func(inputs, targets)
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self._loss_fn(outputs, targets)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.CFG["lr"])