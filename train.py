import pytorch_lightning as pl
from model import Model


def train(CFG):
    model = Model(CFG)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)