from pathlib import Path
import logging
import torch

from data import data_loaders, load_data, clean_data, dictionary
from utils.read_config import reading_config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.setup_logging import setup_logging
from pretrained_embeddings import load_pretrained_embeddings
from config import Config
from models.embeddings_bag import TextClassifier
from optimizer import adam_optimizer
from loss_function import cross_entropy
from stock_prediction import StockPrediction

logger = logging.getLogger("stock-prediction")

def main():
    #output directory
    output_dir = Path('../output/')

    #setup logging
    output_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = Path(output_dir / "output.log")
    setup_logging(logfile = logfile_path)

    #reading the config file
    config_file = Path('../config.ini')
    reading_config(config_file)

    #dataset paths
    rworldnews_path = Path(Config.get("rworldnews"))
    millionnews_path = Path(Config.get("millionnews"))

    #loading the dataset
    raw_data = load_data(rworldnews_path)
    #raw_data = load_data(millionnews_path)

    dates, labels, news = clean_data(raw_data)
    id_to_word, word_to_id = dictionary(news, threshold = 5)
    training_loader, validation_loader, testing_loader = data_loaders(rworldnews_path)

    #tensorboard

    #loading pretrained embeddings
    pretrained_emb_file = Path(Config.get("pretrained_emb_path"))
    pretrained_embeddings, emb_dim = load_pretrained_embeddings(pretrained_emb_file, id_to_word)

    #text classification model
    num_classes = 2
    model = TextClassifier(pretrained_embeddings, emb_dim, num_classes)

    #load the optimizer
    learning_rate = Config.get("learning_rate")
    optimizer = adam_optimizer(model, learning_rate)

    #load the loss function
    criterion = cross_entropy

    #load checkpoint
    checkpoint_file = Path(output_dir / Config.get("checkpoint_file"))
    checkpoint_stocks = load_checkpoint(checkpoint_file)

    #using available device(gpu/cpu)
    model = model.to(Config.get("device"))
    pretrained_embeddings = pretrained_embeddings.to(Config.get("device"))

    #intializing the model and optimizer from the save checkpoint.
    start_epoch = 1
    if checkpoint_stocks is not None:
        start_epoch = checkpoint_stocks['epoch'] + 1
        model.load_state_dict(checkpoint_stocks['model'])
        optimizer.load_state_dict(checkpoint_stocks['optimizer'])
        logger.info('Initialized model and the optimizer from loaded checkpoint.')
    
    del checkpoint_stocks

    #stock prediction model
    model = StockPrediction(model, optimizer, criterion, 
                            training_loader, validation_loader, testing_loader, output_dir)

    #training and testing the model
    epochs = Config.get("epochs")
    validate_every = Config.get("validate_every")
    model.train(epochs, validate_every, start_epoch)



if __name__ == "__main__":
    main()