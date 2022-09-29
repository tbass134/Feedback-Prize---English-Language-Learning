import utils
import pandas as pd
import pytorch_lightning as pl
import train

CFG = {
    'train_path': 'data/train.csv',
    'test_path': 'data/test.csv',
    'model_name': 'microsoft/deberta-base',
    'max_len': 512,
    'train_batch_size': 32,
    'valid_batch_size': 32,
    'dropout': 0.3,
    'num_classes': 6,
    'lr': 1e-5,
    'epochs': 1

}
if __name__ == "__main__":
    utils.download_dataset("data", "kaggle competitions download -c feedback-prize-english-language-learning", "feedback-prize-english-language-learning.zip")

    # train = pd.read_csv("data/train.csv")
    # print(train.head())
    # test = pd.read_csv("data/test.csv") 

    train.train(CFG)

