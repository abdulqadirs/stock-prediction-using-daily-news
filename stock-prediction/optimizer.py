import torch

def adam_optimizer(model, learning_rate):
    """
    Returns the Adam Optimizer.
    
    Args:
        model (object): Text classifier.
        learning_rate (float): Step size of optimizer.
    
    Returns:
        The Adam Optimizer.
    """
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    return optimizer