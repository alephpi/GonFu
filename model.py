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
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model

def build_generator(z_dim,):
  z_rand = Input(shape=(z_dim,))
  x = Dense(32)(z_rand)
  x = LeakyReLU(alpha=0.2)(x)
  x = Dense(32)(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = Dense(6)(x)
  output = x
  model_generator = Model(z_rand, output)
  return model_generator

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
    latent_dim = 50
    latent_variable = noise[:, :latent_dim]

    # load my parameters
    generator = build_generator(z_dim=50)
    generator.load_weights('parameters/weights_generator_GAN')

    return generator(latent_variable)

# test = pd.read_csv('data/df_train.csv')
# test = np.array(test)[7700:, :]
# generated = generative_model(noise=np.random.randn(test.shape[0], 50))

# ax = [None] * 6
# plt.figure(figsize= (20,10))
# for i in range(1,7,1):
#   ax[i-1] = plt.subplot(2,3,i)
#   ax[i-1].hist(test[:,i-1], alpha=0.5, bins=100)
#   ax[i-1].hist(generated[:,i-1], alpha=0.5, bins=100)
# plt.show()
