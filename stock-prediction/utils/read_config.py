import configparser
import logging
import torch

from config import Config

logger = logging.getLogger('stock-prediction')

def reading_config(file_path):
    """
    Reads the config settings from config file and makes them accessible to the project using config.py module.
    Args:
        file_path (Path): The path of config file.
    
    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    # TODO (aq): Raise Error if file doesn't exist. 
    config = configparser.ConfigParser()
    try:
        config.read(file_path)
        logger.info('Reading the config file from: %s' % file_path)
    except FileNotFoundError:
        logger.exception("Config file doesn't exist.")

    #GPUs
    Config.set("disable_cuda", config.getboolean("GPU", "disable_cuda", fallback=False))
    if not Config.get("disable_cuda") and torch.cuda.is_available():
        Config.set("device", "cuda")
        logger.info('GPU is available')
    else:
        Config.set("device", "cpu")
        logger.info('Only CPU is available')

    #paths
    Config.set("rworldnews", config.get("paths", "rworldnews", fallback="datasets/subreddit-worldnews-DJIA-dataset/combined-rworldnews-DJIA.txt"))
    Config.set("millionnews", config.get("paths", "millionnews", fallback="datasets/million-news-headlines-S&P-ASX200-dataset/combined-news-stock.txt"))
    Config.set("output_dir", config.get("paths", "outdir", fallback="output"))
    Config.set("checkpoint_file", config.get("paths", "checkpoint_file", fallback="checkpoint.ImageCaptioning.pth.tar"))

    #pretrained_embeddings
    Config.set("pretrained_emb_path", config.get("pretrained_embeddings", "path", fallback="glove.6B.50d.word2vec.txt"))
    Config.set("pretrained_emb_dim", config.getint("pretrained_embeddings", "emb_dim", fallback=50))

    #Training
    Config.set("training_batch_size", config.getint("training", "batch_size", fallback=4))
    Config.set("epochs", config.getint("training", "epochs", fallback=1000))
    Config.set("learning_rate", config.getfloat("training", "learning_rate", fallback=0.01))

    #Validation
    Config.set("validation_batch_size", config.getint("validation", "batch_size", fallback=4))
    Config.set("validate_every", config.getint("validation", "validate_every", fallback=10))

    #Testing
    Config.set("testing_batch_size", config.getint("testing", "batch_size", fallback=1))