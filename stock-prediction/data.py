import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import itertools
import numpy as np
import torch.utils.data as utils
from PIL import Image
import torch
import PIL.ImageOps
from torch.utils.data import Dataset, DataLoader  
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import pandas as pd

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from config import Config

logger = logging.getLogger('stock-prediction')

def load_data(path):
    """
    Reads the combined news headlines &  the stock exchange data.

    Args:
        path (Path): Path of the file containing the data.

    Returns:
        Pandas dataframe (Date Label News).
    
    Raises:
        FileNotFoundError: If the file containing news-stock data doesn't exist.
    """
    try:
        data = pd.read_csv(path, sep='\t')
    except FileNotFoundError:
        logger.exception("Traceback of data file '{}' not found.".format(path))
    else:
        return data


def clean_data(raw_data):
    """
    Performs following functions: converts sentences to words, converts the words to lower case, 
    removes non english alphabatic characters, etc

    Args:
        raw_data (Pandas Dataframe): Raw data in pandas dataframe (Date Label News)

    Returns:
        List of dates.
        Dictionary of labels ({date: label}).
        Dictionary of list of news headlines ({date: [[headline-1], [headline-2], ... [headline-n]]}).
    """
    all_news = {}
    labels = []
    dates = []
    for _, row in raw_data.iterrows():
        news = row['News']
        date = row['Date']
        label = row['Label']
        headlines = news.split("<.>")
        daily_headlines = []
        for headline in headlines:
            headline = headline.strip('"b')
            #converting the sentences into words
            tokens = word_tokenize(headline)
            #converting to lowercas
            tokens = [w.lower() for w in tokens]
            #remvoing non english alphabetic character
            words = [word for word in tokens if word.isalpha()]
            #removing the stop words
            #words = [w for w in words if not w in stop_words]
            #print(words)
            daily_headlines.append(words)

        dates.append(date)
        labels.append(label)
        all_news[date] = daily_headlines

    #dates = np.array(dates)
    #labels = np.array(labels)
    
    return dates, labels, all_news

def dictionary(cleaned_data,threshold):
    """
    Constructs the dictionary of words in news headlines based on frequency of each word.
    
    Args:
        cleaned_data (dict): Dictionary of {dates: [news headlines]}.
        threshold (int): Words from image captions are being included in dictionary if frequency of words >= threshold.

    Returns:
            id_to_word (list): list of words in dictionary indexed by id.
            word_to_id (dict): dictionary of {word: id}.
    """
    news = []
    for date in cleaned_data:
        for headlines in cleaned_data[date]:
            news.append(headlines)

    word_freq = nltk.FreqDist(itertools.chain(*news))
    id_to_word = ['<pad>'] + [word for word, cnt in word_freq.items() if cnt >= threshold] + ['<unk>']
    word_to_id = {word:idx for idx, word in enumerate(id_to_word)}
    
    return id_to_word, word_to_id


def tokenization(news, word_to_id):
    """
    Represents the raw captions by list of ids in the dictionary

    Args:
        news (dict): A dictionary of {date: list of news headlines}.
        word_to_id (dict): A dictionary of {word: id}.

    Returns:
        tokenized_news(list): List of ids of word from dictionary.
        dates(list): List of dates.
        lengths(list): Actual length of each news healdine.
    """
    tokenized_news = []
    lengths = []
    for date in news:
        daily_headlines = []
        daily_lengths = []
        for headline in news[date]:
            daily_lengths.append(len(headline))
            token = []
            for word in headline:
                if word in word_to_id:
                    token.append(word_to_id[word])
                else:
                    token.append(word_to_id['<unk>'])
            daily_headlines.append(token)

        lengths.append(daily_lengths)
        tokenized_news.append(daily_headlines)
    #tokens = np.array(tokens).astype('int32')
    return tokenized_news, lengths


class PadSequence(object):
    """
    Pads the unequal sequences with zeros in a batch to make them equal to the largest sequence in the batch.
    """
    def __call__(self, batch):
        dates = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        news = [item[2] for item in batch]
        lengths = [item[3] for item in batch]
        #calculating the sum of lengths of dialy headline
        lengths_sum = []
        for headline_length in lengths:
            lengths_sum.append(sum(headline_length))

        news_batch = []
        for daily_news in news:
            concatenated_daily_news = []
            for headline in daily_news:
                for word in headline:
                    concatenated_daily_news.append(word)
            news_batch.append(concatenated_daily_news)

        
        max_length = max(lengths_sum)
        if len(lengths_sum) > 1:
            for daily_news in news_batch:
                if len(daily_news) < max_length:
                    pad = [0] * (max_length - len(daily_news))
                    daily_news += pad
        
        return dates, torch.tensor(labels), torch.tensor(news_batch), torch.tensor(lengths)

        
class NewsStockDataLoader(Dataset):
    """
    Loads the dataset.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.raw_data = load_data(self.dataset_path)
        self.dates, self.labels, self.news = clean_data(self.raw_data)
        self.id_to_word, self.word_to_id = dictionary(self.news, threshold = 5)
        self.tokenized_news, self.lengths = tokenization(self.news, self.word_to_id)

    def __getitem__(self, index):
        dates = self.dates[index]
        labels = self.labels[index]
        news = self.tokenized_news[index]
        lengths = self.lengths[index]

        news = news[0:10]
        lengths = lengths[0:10]
        return dates, labels, news, lengths

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.dates)


def data_loaders(dataset_path):
    """
    Loads the data and divides it into training, validation and test sets.

    Args:
        dataset_path (Path): Path of news-stock dataset.

    Returns:
        training_data_loader
        validation_data_loader
        testing_data_loader
    """
    dataset_path = dataset_path
    news_stock_dataset = NewsStockDataLoader(dataset_path)
    
    dataset_size = len(news_stock_dataset)
    indices = list(range(dataset_size))
    training_split = int(0.8 * dataset_size)
    validation_split = int(0.9 * dataset_size)

    np.random.seed(96)
    np.random.shuffle(indices)

    train_indices = indices[:training_split]
    valid_indices = indices[training_split:validation_split]
    test_indices = indices[validation_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    collate = PadSequence()

    training_loader = DataLoader(news_stock_dataset,
                        num_workers = 1,
                        batch_size = Config.get("training_batch_size"),
                        sampler = train_sampler,
                        collate_fn = collate)

    validation_loader = DataLoader(news_stock_dataset,
                        num_workers = 1,
                        batch_size = Config.get("validation_batch_size"),
                        sampler = valid_sampler,
                        collate_fn = collate)

    testing_loader = DataLoader(news_stock_dataset,
                        num_workers = 1,
                        batch_size = Config.get("testing_batch_size"),
                        sampler= test_sampler,
                        collate_fn = collate)
    
    return training_loader, validation_loader, testing_loader