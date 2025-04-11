
import torch
import torch.nn as nn
class VariationalAutoEncoder(nn.Module):

  def __init__(self, in_channels):

    super(VariationalAutoEncoder,self).__init__()

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

    self.Conv3_Encoder = nn.Sequential(
        nn.Conv2d(in_channels = 150,
                  out_channels = 150,
                  kernel_size = 3,
                  padding = 1,
                  stride = 1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features = 150)
    )

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

    self.Encoder = nn.Sequential(
        self.Conv1_Encoder,
        self.Conv2_Encoder,
        self.Conv3_Encoder,
        self.Conv4_Encoder,
        self.Conv5_Encoder,
        nn.Flatten()
    )

    self.mean = nn.Linear(200 * 4 * 4, 30)
    self.log_var = nn.Linear(200 * 4 * 4, 30)

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

    self.Conv3_Decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels =  150,
                           out_channels = 150,
                           kernel_size = 3,
                           padding = 1,
                           stride = 1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features = 150)
    )


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

    self.flatten = nn.Flatten()

  def reparameterization_trick(self, mean, log_var):

    epsilon = torch.randn_like(log_var).to(mean.device)

    z = mean + log_var * epsilon

    return z

  def forward(self,x):


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
