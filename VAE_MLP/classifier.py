import torch.nn as nn

# Defining the structure of a classifier

class Classifier(nn.Module):

  def __init__(self, in_features):

    super(Classifier,self).__init__()

    # Block of fully Connected layers for classification

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
