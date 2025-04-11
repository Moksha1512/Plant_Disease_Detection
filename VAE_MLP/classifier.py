import torch.nn as nn

class Classifier(nn.Module):

  def __init__(self, in_features):

    super(Classifier,self).__init__()
    self.FullyConnectedBlock = nn.Sequential(
        nn.Linear(in_features,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,15)
    )

  def forward(self,x):

    output = self.FullyConnectedBlock(x)

    return output
