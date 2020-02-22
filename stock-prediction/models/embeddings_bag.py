import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, pretrained_emb, emb_dim, num_classes):
        super().__init__()
        self.pretrained_emb = pretrained_emb
        self.fc = nn.Linear(emb_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, text, offsets):
        embedded = self.pretrained_emb(text, offsets)
        output = self.fc(embedded)
        ouput_prob = self.softmax(output)
        return ouput_prob