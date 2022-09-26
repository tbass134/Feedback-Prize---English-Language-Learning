import pytorch_lightning as pl
from model import Model


def train(train_path, test_path):
    model = Model(train_path,test_path)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)