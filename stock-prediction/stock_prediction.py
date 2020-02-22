import torch
import numpy as np
import logging
from tqdm import tqdm

from config import Config
from utils.checkpoint import save_checkpoint

logger = logging.getLogger("stock-prediction")

class StockPrediction:
    """
    Runs the model on training, validation and test datasets.

    Attributes:
        model (object): Text classification model.
        optimizer (object): Gradient descent optimizer.
        criterion (object): Loss function.
        training_loader:
        validation_loader:
        testing_loader:
        output_dir:
    """
    def __init__(self, model, optimizer, criterion, 
                training_loader, validation_loader, testing_loader,
                output_dir):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.testing_loader = testing_loader
        self.output_dir = output_dir

    def train(self, epochs, validate_every, start_epoch):
        """
        Runs the model on the training dataset.

        Args:
            epochs (int): Total epochs.
            validate_every (int): Run validation after every validate_every no of epochs.
            start_epoch (int): Starting epoch if using the stored checkpoint. 
        """    
        for epoch in range(start_epoch, epochs + 1):
            training_batch_losses = []
            for _, data in tqdm(enumerate(self.training_loader, 0)):
                dates, labels, news, lengths_sum = data
                self.optimizer.zero_grad()
                #setting up training mode
                self.model = self.model.train()

                offsets = [0]
                for daily_headlines in news:
                    offsets.append(daily_headlines.size(0))
                offsets = torch.tensor(offsets[:-1]).cumsum(0)
                news = news.view(1, -1).squeeze(0)
                target_labels = torch.tensor([int(label.item()) for label in labels])

                news = news.to(Config.get("device"))
                offsets = offsets.to(Config.get("device"))
                target_labels = target_labels.to(Config.get("device"))

                #running the model to calculate the predicted labels
                predicted_labels = self.model(news, offsets)
                #calculating the loss
                loss = self.criterion(predicted_labels, target_labels)
                #calculating the gradients
                loss.backward()
                #updating the parameters
                self.optimizer.step()
                training_batch_losses.append(loss.item())

            if (epoch - 1) % validate_every == 0:
                self.validation(epoch = epoch)
                save_checkpoint(epoch = epoch,
                                outdir = self.output_dir,
                                model = self.model,
                                optimizer = self.optimizer)

            print('Epoch: ', epoch)
            print(' Training Loss: ', np.mean(training_batch_losses))

    
    def validation(self, epoch = None):
        """
        Runs the model on validation dataset.

        Args:
            epoch (int): Current epoch.
        """
        #batch_size = Config.get("validation_batch_size")
        validation_loss = []
        for _, data in tqdm(enumerate(self.validation_loader, 0)):
                _, labels, news, _ = data
                self.optimizer.zero_grad()
                #setting up training mode
                self.model = self.model.train()

                offsets = [0]
                for daily_headlines in news:
                    offsets.append(daily_headlines.size(0))
                offsets = torch.tensor(offsets[:-1]).cumsum(0)  
                news = news.view(1, -1).squeeze(0)
                target_labels = torch.tensor([int(label.item()) for label in labels])

                news = news.to(Config.get("device"))
                offsets = offsets.to(Config.get("device"))
                target_labels = target_labels.to(Config.get("device"))

                #running the model to calculate the predicted labels
                predicted_labels = self.model(news, offsets)
                #calculating the loss
                loss = self.criterion(predicted_labels, target_labels)

                validation_loss.append(loss.item())
        
        print("Validation Loss: ", np.mean(validation_loss))

    def testing(self):
        """
        Runs the model on test dataset.
        """
        testing_loss = []
        for _, data in tqdm(enumerate(self.testing_loader, 0)):
                _, labels, news, _ = data
                self.optimizer.zero_grad()
                #setting up training mode
                self.model = self.model.train()

                offsets = [0]
                for daily_headlines in news:
                    offsets.append(daily_headlines.size(0))
                offsets = torch.tensor(offsets[:-1]).cumsum(0)  
                news = news.view(1, -1).squeeze(0)
                target_labels = torch.tensor([int(label.item()) for label in labels])

                news = news.to(Config.get("device"))
                offsets = offsets.to(Config.get("device"))
                target_labels = target_labels.to(Config.get("device"))

                #running the model to calculate the predicted labels
                predicted_labels = self.model(news, offsets)
                #calculating the loss
                loss = self.criterion(predicted_labels, target_labels)

                testing_loss.append(loss.item())
        
        print("Testing Loss: ", np.mean(testing_loss))