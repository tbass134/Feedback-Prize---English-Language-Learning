import utils
import pandas as pd
import pytorch_lightning as pl
import train

CFG = {
    'train_path': 'data/train.csv',
    'test_path': 'data/test.csv',
    # 'model_name': 'microsoft/deberta-base',
    'model_name': 'bert-base-uncased',
    'max_len': 512,
    'train_batch_size': 16,
    'valid_batch_size': 16,
    'dropout': 0.5,
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

