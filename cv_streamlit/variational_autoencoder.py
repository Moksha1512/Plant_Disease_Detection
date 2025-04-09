
  # Defining the structure of a variational autoencoder
import torch
import torch.nn as nn
class VariationalAutoEncoder(nn.Module):

  def __init__(self, in_channels):

    super(VariationalAutoEncoder,self).__init__()

    # Convolution Blocks for the Encoder

    # Padding = (Kernel Size - 1) / 2
    # Kernel Size = 3
    # Padding = (3 - 1) / 2 = 1

    # Dimensions [Conv1_Encoder]
    # Input             : (-1,  3, 64, 64)
    # After Conv2d      : (-1,100, 64, 64)
    # After MaxPool2d   : (-1,100, 32, 32)

    self.Conv1_Encoder = nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                  out_channels = 100,
                  kernel_size = 3,
                  padding = 1,
                  stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.BatchNorm2d(num_features = 100)
    )

    # Dimensions [Conv2_Encoder]
    # Input             : (-1,100, 32, 32)
    # After Conv2d      : (-1,150, 32, 32)
    # After MaxPool2d   : (-1,150, 16, 16)

    self.Conv2_Encoder = nn.Sequential(
        nn.Conv2d(in_channels = 100,
                  out_channels = 150,
                  kernel_size = 3,
                  padding = 1,
                  stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.BatchNorm2d(num_features = 150)
    )

    # Dimensions [Conv3_Encoder]
    # Input             : (-1,150, 16, 16)
    # After Conv2d      : (-1,150, 16, 16)
    # After MaxPool2d   : (-1,150, 16, 16)

    self.Conv3_Encoder = nn.Sequential(
        nn.Conv2d(in_channels = 150,
                  out_channels = 150,
                  kernel_size = 3,
                  padding = 1,
                  stride = 1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features = 150)
    )

    # Dimensions [Conv4_Encoder]
    # Input             : (-1,150, 16, 16)
    # After Conv2d      : (-1,200, 16, 16)
    # After MaxPool2d   : (-1,200,  8,  8)

    self.Conv4_Encoder = nn.Sequential(
        nn.Conv2d(in_channels = 150,
                  out_channels = 200,
                  kernel_size = 3,
                  padding = 1,
                  stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.BatchNorm2d(num_features = 200)
    )

    # Dimensions [Conv5_Encoder]
    # Input             : (-1,200,  8,  8)
    # After Conv2d      : (-1,200,  8,  8)
    # After MaxPool2d   : (-1,200,  4,  4)

    self.Conv5_Encoder = nn.Sequential(
        nn.Conv2d(in_channels = 200,
                  out_channels = 200,
                  kernel_size = 3,
                  padding = 1,
                  stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.BatchNorm2d(num_features = 200)
    )

    # Encoder

    # Dimensions [Encoder]
    # Input                    : (-1,   3, 64, 64)
    # After Conv1_Encoder      : (-1, 100, 32, 32)
    # After Conv2_Encoder      : (-1, 150, 16, 16)
    # After Conv3_Encoder      : (-1, 150, 16, 16)
    # After Conv4_Encoder      : (-1, 200,  8,  8)
    # After Conv5_Encoder      : (-1, 200,  4,  4)
    # After Flatten            : (-1, 200 * 4 * 4)

    self.Encoder = nn.Sequential(
        self.Conv1_Encoder,
        self.Conv2_Encoder,
        self.Conv3_Encoder,
        self.Conv4_Encoder,
        self.Conv5_Encoder,
        nn.Flatten()
    )

    # Latent Space

    self.mean = nn.Linear(200 * 4 * 4, 30)
    self.log_var = nn.Linear(200 * 4 * 4, 30)

    # Convolution Blocks for the Decoder

    # Padding = (Kernel Size - 1) / 2
    # Kernel Size = 3
    # Padding = (3 - 1) / 2 = 1

    # Dimensions [Conv1_Decoder]
    # Input                      : (-1, 200,  4,  4)
    # After ConvTranspose2d      : (-1, 200,  8,  8)

    self.Conv1_Decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels =  200,
                           out_channels =  200,
                           kernel_size = 3,
                           padding = 1,
                           output_padding = 1,
                           stride = 2),
        nn.ReLU(),
        nn.BatchNorm2d(num_features = 200)
    )

    # Dimensions [Conv2_Decoder]
    # Input                      : (-1, 200,  8,  8)
    # After ConvTranspose2d      : (-1, 150, 16, 16)

    self.Conv2_Decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels =  200,
                           out_channels = 150,
                           kernel_size = 3,
                           padding = 1,
                           output_padding = 1,
                           stride = 2),
        nn.ReLU(),
        nn.BatchNorm2d(num_features = 150)
    )

    # Dimensions [Conv3_Decoder]
    # Input                      : (-1, 150, 16, 16)
    # After ConvTranspose2d      : (-1, 150, 16, 16)

    self.Conv3_Decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels =  150,
                           out_channels = 150,
                           kernel_size = 3,
                           padding = 1,
                           stride = 1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features = 150)
    )

    # Dimensions [Conv4_Decoder]
    # Input                      : (-1, 150, 16, 16)
    # After ConvTranspose2d      : (-1, 100, 32, 32)

    self.Conv4_Decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels =  150,
                           out_channels = 100,
                           kernel_size = 3,
                           padding = 1,
                           output_padding = 1,
                           stride = 2),
        nn.ReLU(),
        nn.BatchNorm2d(num_features = 100)
    )

    # Dimensions [Conv5_Decoder]
    # Input                      : (-1, 100, 32, 32)
    # After ConvTranspose2d      : (-1,   3, 64, 64)

    self.Conv5_Decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels =  100,
                           out_channels = in_channels,
                           kernel_size = 3,
                           padding = 1,
                           output_padding = 1,
                           stride = 2),
        nn.ReLU(),
        nn.BatchNorm2d(num_features = in_channels)
    )

    # Decoder

    # Dimensions [Decoder]
    # Input                    : (-1,          30)
    # After Linear             : (-1, 200 * 4 * 4)
    # After Unflatten          : (-1, 200,  4,  4)
    # After Conv1_Decoder      : (-1, 200,  8,  8)
    # After Conv2_Decoder      : (-1, 150, 16, 16)
    # After Conv3_Decoder      : (-1, 150, 16, 16)
    # After Conv4_Decoder      : (-1, 100, 32, 32)
    # After Conv5_Decoder      : (-1,   3, 64, 64)

    self.Decoder = nn.Sequential(
        nn.Linear(30, 200*4*4),
        nn.Unflatten(1, (200,4,4)),
        self.Conv1_Decoder,
        self.Conv2_Decoder,
        self.Conv3_Decoder,
        self.Conv4_Decoder,
        self.Conv5_Decoder,
        nn.Sigmoid()
    )

    # For Flattening

    self.flatten = nn.Flatten()

  def reparameterization_trick(self, mean, log_var):

    # Defining the reparameterization trick

    epsilon = torch.randn_like(log_var).to(mean.device)

    z = mean + log_var * epsilon

    return z

  def forward(self,x):

    # Passing the input sequentially through each of the above-defined blocks

    x = self.Conv1_Encoder(x)
    x_r = self.Conv2_Encoder(x)
    x = self.Conv3_Encoder(x_r)
    x = self.Conv4_Encoder(x + x_r)
    x = self.Conv5_Encoder(x)
    x = self.flatten(x)

    mean = self.mean(x)
    log_var = self.log_var(x)

    encoded = self.reparameterization_trick(mean, torch.exp(0.5 * log_var))

    decoded = self.Decoder(encoded)

    return encoded, decoded, mean, log_var