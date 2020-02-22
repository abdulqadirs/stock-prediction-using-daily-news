import torch
from pathlib import Path
import logging
from config import Config

logger = logging.getLogger("stock-prediction")

def save_checkpoint(epoch, outdir, model, optimizer, filename = 'checkpoint.ImageCaptioning.pth.tar'):
    """
    Saves the model checkpoint
    Args:
        epoch (int): current epoch
        outdir (Path): directory to output checkpoint
        model (object): stock prediction model.
        optimizer (object): optimizer for the model
        filename (Path): for checkpoint in outdir
    """
    filename = outdir / filename
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, str(filename))


def load_checkpoint(checkpoint_file):
    """
    Loads the checkpoint of the epoch, model, optimizer
    
    Args:
        checkpoint_file (Path): file name of latest checkpoint file
    Returns:
        checkpoint (dict):
    
    Raises:
        warning: If the checkpoint file doesn't exist.
    """
    checkpoint = None
    try:
        checkpoint = torch.load(checkpoint_file, map_location=Config.get("device"))
        logger.info('Loading the checkpoint file')
    except:
        logger.warning('Checkpoint file does not exist')
    
    return checkpoint