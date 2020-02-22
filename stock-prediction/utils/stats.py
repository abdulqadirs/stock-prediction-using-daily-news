import logging

logger = logging.getLogger("stock-prediction")

class Statistics:
    """
    Records the losses and evaluation metrics.
    Writes the losses and evaluation metrics to tensorboard.
    Logs the losses and evaluation metrics.

    Attributes:
        outdir (Path): Output directory to save tensorboard log files.
        tensorboard_writer (object):

    """
    def __init__(self, outdir, tensorboard_writer):
        self.outdir = outdir
        self.tensorboard_writer = tensorboard_writer
        self.training_losses = []
        self.validation_losses = []
        self.testing_losses = []
    

    def record(self, training_losses=None, validation_losses=None, testing_losses=None):
        """
        Stores the statistics in lists.
        """
        self.training_losses.append(training_losses) if training_losses is not None else {}
        self.validation_losses.append(validation_losses) if validation_losses is not None else {}
        self.testing_losses.append(testing_losses) if testing_losses is not None else {}


    def log_losses(self, epoch):
        """
        Pushes the loss of given epoch to stout and logfile.
        """
        logger.info("At epoch {}. Train Loss: {}".format(epoch, self.training_losses[-1]))
    
    
    def log_eval(self, epoch, dataset_name):
        """
        Outputs the evaluation metrics score to stout and logfile.
        """
        pass

    def push_tensorboard_losses(self, epoch):
        """
        Pushes the losses to tensorboard.
        """
        if self.training_losses:
            self.tensorboard_writer.add_scalar('losses/train', self.training_losses[-1], epoch)
        if self.validation_losses:
            self.tensorboard_writer.add_scalar('losses/validation', self.validation_losses[-1], epoch)
        if self.testing_losses:
            self.tensorboard_writer.add_scalar('losses/testing', self.testing_losses[-1], epoch)
    

    def push_tensorboard_eval(self, epoch, dataset_name):
        """
        Pushes the evaluation metrics score to tensorboard.
        """
        pass