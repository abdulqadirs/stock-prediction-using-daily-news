import torch.nn as nn

def cross_entropy(predicted_labels, target_labels):
    """
    Calcualtes the cross entropy loss function using pytorch.
    
    Args:
        predicted_captions (tensor): label predicted by model of shape(batch_size, num_clases).
        target_captions (tensor): reference label of shape(batch_size, num_classes).
    
    Returns:
        Cross entory loss of shape(batch_size).
    """
    loss = nn.CrossEntropyLoss()
    # batch_size, captions_length, vocab_length = predicted_labelss.size()
    # predicted_labels = predicted_labelss.view(batch_size * captions_length, -1)
    # target_captions = target_labels.view(-1)
    error = loss(predicted_labels, target_labels)

    return error