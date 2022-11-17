##############################################
# YOU MUST NOT MODIFY THIS FILE
##############################################
from model import generative_model
import numpy as np
import logging


logging.basicConfig(filename="check.log", level=logging.DEBUG, 
                    format="%(asctime)s:%(levelname)s: %(message)s", 
                    filemode='w')

def simulate(noise):
    """
    simulation of your Generative Model

    Parameters
    ----------
    model : object
        generator function
    noise : ndarray
        input of the generative model
    """

    try:
        output = generative_model(noise)
        message = "Successful simulation" 
        assert output.shape == (noise.shape[0], 6), "Shape error, it must be (n_data, 6). Please verify the shape of the output."
        
        # write the output
        np.save("output.npy", output)

    except Exception as e:
        message = e
                
    finally:
        logging.debug(message)

    return output

    
if __name__ == "__main__":
    noise = np.load("data/noise.npy")
    simulate(noise)
    
    
