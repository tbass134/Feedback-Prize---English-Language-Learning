import utils
import pandas as pd
import pytorch_lightning as pl
import train
if __name__ == "__main__":
    utils.download_dataset("data", "kaggle competitions download -c feedback-prize-english-language-learning", "feedback-prize-english-language-learning.zip")

    # train = pd.read_csv("data/train.csv")
    # print(train.head())
    # test = pd.read_csv("data/test.csv") 

    train.train(train_path="data/train.csv", test_path="data/test.csv")

