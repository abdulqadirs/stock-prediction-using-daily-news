from pathlib import Path
import torch

from data import data_loaders
from utils.read_config import reading_config

config_file = Path('../config.ini')
reading_config(config_file)

path = '../datasets/subreddit-worldnews-DJIA-dataset/combined-rworldnews-DJIA.txt'
training_loader, validation_loader, testing_loader = data_loaders(path)
for _, data in enumerate(training_loader, 0):
    dates, labels, news, lengths = data
    print("News: ", news)
    print("lengths", lengths)
    print("labels: ", labels)
    #print("dates: ", dates)
    for n in news:
        print(len(n))
    break

