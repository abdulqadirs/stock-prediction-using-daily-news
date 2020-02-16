from pathlib import Path
import torch

from data import data_loaders, load_data, clean_data, dictionary
from utils.read_config import reading_config
from pretrained_embeddings import load_pretrained_embeddings
from config import Config
from models.embeddings_bag import TextClassifier

config_file = Path('../config.ini')
reading_config(config_file)

#data path
rworldnews_path = Path(Config.get("rworldnews"))

#loading the data
raw_data = load_data(rworldnews_path)
dates, labels, news = clean_data(raw_data)
id_to_word, word_to_id = dictionary(news, threshold = 5)

#loading pretrined embeddings
pretrained_emb_file = Path(Config.get("pretrained_emb_path"))
pretrained_embedddings, emb_dim = load_pretrained_embeddings(pretrained_emb_file, id_to_word)

training_loader, validation_loader, testing_loader = data_loaders(rworldnews_path)

#embedding_bag
text_classifier = TextClassifier(pretrained_embedddings, emb_dim, 2)

for _, data in enumerate(training_loader, 0):
    dates, labels, news, lengths_sum = data
    offsets = [0]
    for daily_headlines in news:
        offsets.append(daily_headlines.size(0))
    offsets = torch.tensor(offsets[:-1]).cumsum(0)  
    news = news.view(1, -1).squeeze(0)
    
    output = text_classifier(news, offsets)
    print("output shape: ", output.shape)
    print("output: ", output)
    # print("Dates: ", dates)
    # print("labels: ", labels)
    # print("News shape: ", news.shape)
    # print("News: ", news)
    # print("lengths sum", lengths_sum)
    # print("offsets: ", offsets)
    break

