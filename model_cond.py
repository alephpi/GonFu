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

from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
import tensorflow

def build_generator(z_dim,):

  z_rand = Input(shape=(z_dim,))
  x = Dense(32)(z_rand)
  x = LeakyReLU(alpha=0.2)(x)
  x = Dense(16)(x)
  x = LeakyReLU(alpha=0.2)(x)
  x = Dense(6)(x)
  mean = np.array([-0.00375393,  0.01648988, -0.01080703,  0.00253291,  0.00089046,
        0.00957318])
  mean = tensorflow.convert_to_tensor(mean)
  mean = tensorflow.cast(mean, tensorflow.float32)
  output = x - mean
  model_generator = Model(z_rand, output)
  model_generator.summary()

  return model_generator

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise, position):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    latent_variable = noise[:, :50]  # use the first 10 dimensions of the noise
    
    # load my parameters (of dimension 10 in this example). 
    # <!> be sure that they are stored in the parameters/ directory <!>
    generator = build_generator(z_dim=50)
    generator.load_weights('parameters/weights_generator')


    return generator(latent_variable)



