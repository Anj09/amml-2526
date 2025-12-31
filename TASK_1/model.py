import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.capacity = 64
        self.latent_dims = 20
        # encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.capacity, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=self.capacity, out_channels=self.capacity*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=self.capacity*2*7*7, out_features=self.latent_dims)
        self.fc_logvar = nn.Linear(in_features=self.capacity*2*7*7, out_features=self.latent_dims)
        # decoder
        self.fc_decode = nn.Linear(in_features=self.latent_dims, out_features=self.capacity*2*7*7)
        self.conv2_decode = nn.ConvTranspose2d(in_channels=self.capacity*2, out_channels=self.capacity, kernel_size=4, stride=2, padding=1)
        self.conv1_decode = nn.ConvTranspose2d(in_channels=self.capacity, out_channels=1, kernel_size=4, stride=2, padding=1)

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

    def decoder(self, x):
        x = self.fc_decode(x)
        x = x.view(x.size(0), self.capacity*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2_decode(x))
        x = torch.sigmoid(self.conv1_decode(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        # the reparameterization trick
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(mu)
