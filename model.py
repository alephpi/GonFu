#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# G_\theta(Z) = np.max(0, \theta.Z)
############################################################################

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 6
learning_rate = 5e-4
batch_size = 4

# VAE model
class VAE(nn.Module):
    def __init__(self, input_size=6, h_dim=256, z_dim=20, out_channels=16, kernel_size=3, pool_size=2):
        super(VAE, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=kernel_size, padding='same')
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride = pool_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15*out_channels*2, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_size)
        
    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h) # mu(mean of q(z|x)) & log variance
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std) # generate standard gaussian variable
        return mu + eps * std # make backpropa possible

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_dim = 20
    latent_variable = torch.tensor(noise[:, :latent_dim]).to('cpu')
    # load my parameters (of dimension 10 in this example). 
    parameters = torch.load(os.path.join("parameters", "weights"))
    model = VAE()
    model.load_state_dict(parameters)

    # return np.maximum(0, latent_variable @ parameters)
    return model.decode(latent_variable).detach().numpy()

